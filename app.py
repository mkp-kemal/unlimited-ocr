#!/usr/bin/env python
"""Demo web pembanding tiga engine OCR di Apple Silicon, dengan visual realtime.

Tiga panel: halaman dokumen dengan highlight blok yang sedang dibaca, output mentah
engine, dan Markdown yang tumbuh seiring blok selesai — termasuk potongan gambar
dari dokumen.

Jalankan: .venv/bin/python app.py  ->  http://127.0.0.1:7860
"""

import base64
import io
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image, ImageDraw

from biaya import catat
from field import FIELD_BAWAAN, teks_untuk_llm
from parser_field import cari as cari_field
from parser_field import tata_letak
from llm import MODEL_BAWAAN, MODEL_LLM, PROMPT_ANALISIS, ambil_field, analisis, baca_env
from ocr import (
    IMAGE_EXT,
    MODEL,
    PROMPT,
    SKALA,
    blok_det,
    blok_ke_markdown,
    pdf_ke_gambar,
    perbesar_bila_kecil,
    potong_bbox,
)

AKAR = Path(__file__).resolve().parent
WARNA_AKTIF = (255, 87, 34)
WARNA_SELESAI = (33, 150, 243)
LEBAR_POTONGAN = 420

# Jeda saat memutar blok dari engine batch. Unlimited-OCR benar-benar streaming per
# token; dua lainnya baru mengirim hasil setelah satu halaman rampung, jadi bloknya
# diputar berjarak supaya alur highlight tetap terbaca. Ini tampilan, bukan kecepatan
# asli — statusnya menyebut apa adanya.
JEDA_BLOK = 0.1

MESIN = {
    "Unlimited-OCR — MLX": {
        "jenis": "lokal",
        "catatan": "realtime per token",
    },
    "MinerU — MPS": {
        "jenis": "subproses",
        "python": AKAR.parent / ".venv-mineru/bin/python",
        "skrip": AKAR / "mesin/mineru_runner.py",
        "umpan": "dokumen",
        "catatan": "pipeline deteksi+rekognisi, blok diputar per halaman",
    },
    "MinerU hybrid — VLM": {
        "jenis": "subproses",
        "python": AKAR.parent / ".venv-mineru/bin/python",
        "skrip": AKAR / "mesin/mineru_runner.py",
        "umpan": "dokumen",
        # Isi tabel dibaca VLM, bukan pengenal tabel pipeline. Pada formulir tulisan
        # tangan struktur selnya jauh lebih benar — di SPPA JASINDO, "Nama Lengkap" baru
        # punya nilai lewat jalur ini — tapi ongkosnya ±60 detik/halaman, 7x pipeline.
        "catatan": "isi tabel dibaca VLM, sel lebih benar, ±60 detik/halaman",
        "env": {"MINERU_BACKEND": "hybrid-engine"},
    },
    "PaddleOCR-VL — MLX": {
        "jenis": "subproses",
        "python": AKAR.parent / ".venv-paddle/bin/python",
        "skrip": AKAR / "mesin/paddleocr_runner.py",
        "umpan": "halaman",
        "catatan": "layout di CPU Paddle, pengenalan teks di server mlx-vlm:8111",
        "server": "http://127.0.0.1:8111/v1/models",
    },
}

_state = {}


def muat_model():
    """Model MLX dimuat sekali lalu dipakai ulang; panggilan pertama makan beberapa detik."""
    if "model" not in _state:
        from mlx_vlm import load, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        _state["model"], _state["processor"] = load(MODEL)
        _state["config"] = load_config(MODEL)
        _state["template"] = apply_chat_template
        _state["stream"] = stream_generate
    return _state


def ke_piksel(bbox, size):
    x1, y1, x2, y2 = bbox
    w, h = size
    return x1 / SKALA * w, y1 / SKALA * h, x2 / SKALA * w, y2 / SKALA * h


def gambar_kotak(dasar, kotak):
    """Salin halaman lalu tandai tiap bbox; yang terakhir disorot lebih tebal."""
    im = dasar.convert("RGBA")
    lapis = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lapis)
    tebal = max(2, round(min(im.size) / 300))

    for i, bbox in enumerate(kotak):
        px = ke_piksel(bbox, im.size)
        terakhir = i == len(kotak) - 1
        warna = WARNA_AKTIF if terakhir else WARNA_SELESAI
        d.rectangle(px, fill=warna + (60 if terakhir else 22,))
        d.rectangle(px, outline=warna + (255,), width=tebal * 2 if terakhir else tebal)

    return Image.alpha_composite(im, lapis).convert("RGB")


def pembuat_gambar(pil, cache):
    """Callback blok gambar untuk panel Markdown: potongan halaman jadi data URI."""

    def buat(bbox):
        if bbox not in cache:
            potong = potong_bbox(pil, bbox)
            if min(potong.size) < 8:
                cache[bbox] = None
            else:
                if potong.width > LEBAR_POTONGAN:
                    tinggi = round(potong.height * LEBAR_POTONGAN / potong.width)
                    potong = potong.resize((LEBAR_POTONGAN, tinggi), Image.LANCZOS)
                buf = io.BytesIO()
                potong.save(buf, format="PNG")
                data = base64.b64encode(buf.getvalue()).decode()
                cache[bbox] = f"![blok gambar](data:image/png;base64,{data})"
        return cache[bbox]

    return buat


def pastikan_server(cfg, nama_mesin):
    """Server mlx-vlm mati ikut restart laptop. Tanpa cek ini, galatnya tidak terbaca."""
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(cfg["server"], timeout=3)
    except (urllib.error.URLError, OSError):
        raise gr.Error(
            f"{nama_mesin} butuh server mlx-vlm di {cfg['server']}, tapi tidak merespons. "
            "Nyalakan dulu:  .venv/bin/python -m mlx_vlm.server --port 8111 "
            "--model PaddlePaddle/PaddleOCR-VL-1.6 --trust-remote-code"
        )


def siapkan_halaman(src, dpi, tmpdir):
    if src.suffix.lower() == ".pdf":
        return pdf_ke_gambar(src, int(dpi), None, tmpdir)
    if src.suffix.lower() in IMAGE_EXT:
        return [(1, perbesar_bila_kecil(src, tmpdir))]
    raise gr.Error(f"Format tidak didukung: {src.suffix}")


def baris_mentah(blok):
    return [
        json.dumps({"kategori": k, "bbox": list(b), "teks": t}, ensure_ascii=False)
        for k, b, t in blok
    ]


def jalankan_unlimited(gambar, halaman, max_tokens, prompt, cache):
    """Streaming asli: token demi token, kotak digambar begitu tag <|det|> lengkap."""
    s = muat_model()
    teks_prompt = s["template"](s["processor"], s["config"], prompt, num_images=1)
    selesai, selesai_md = [], []
    t_awal = time.perf_counter()

    for (nomor, img), (_, pil) in zip(gambar, halaman):
        buat_gambar = pembuat_gambar(pil, cache)
        buf, n_kotak, n_token, blok = "", 0, 0, []
        t = time.perf_counter()

        for potong in s["stream"](
            s["model"],
            s["processor"],
            teks_prompt,
            [str(img)],
            max_tokens=int(max_tokens),
            temperature=0.0,
        ):
            buf += potong.text
            n_token += 1
            blok = blok_det(buf)[1]
            kotak = [b for _, b, _ in blok]
            kotak_baru = len(kotak) > n_kotak
            n_kotak = len(kotak)

            # Panel teks disegarkan tiap beberapa token; gambar hanya saat ada kotak
            # baru, karena menyusun ulang overlay jauh lebih mahal daripada kirim teks.
            if not kotak_baru and n_token % 8:
                continue

            lewat = time.perf_counter() - t
            yield (
                gambar_kotak(pil, kotak) if kotak_baru else gr.skip(),
                "\n\n".join(selesai + [f"<!-- halaman {nomor} -->\n{buf}"]),
                "\n\n".join(
                    selesai_md
                    + [f"<!-- halaman {nomor} -->\n\n{blok_ke_markdown(blok, buat_gambar)}"]
                ),
                f"Halaman {nomor}/{len(gambar)} — {n_kotak} blok, {n_token} token, "
                f"{lewat:.1f} s ({n_token / max(lewat, 0.01):.0f} token/s)",
                nomor,
            )

        selesai.append(f"<!-- halaman {nomor} -->\n{buf.strip()}")
        selesai_md.append(
            f"<!-- halaman {nomor} -->\n\n{blok_ke_markdown(blok, buat_gambar)}"
        )

    yield _akhir(selesai, selesai_md, len(gambar), time.perf_counter() - t_awal)


def jalankan_subproses(cfg, src, gambar, halaman, cache):
    """Engine batch: runner di venv-nya sendiri mengirim JSONL, dibaca baris demi baris."""
    umpan = [str(src)] if cfg["umpan"] == "dokumen" else [str(p) for _, p in gambar]
    proses = subprocess.Popen(
        [str(cfg["python"]), str(cfg["skrip"]), *umpan],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        cwd=AKAR,
        env={**os.environ, **cfg.get("env", {})},
    )

    pil_per_halaman = dict(halaman)
    blok_per_halaman, selesai, selesai_md = {}, [], []
    t_awal = time.perf_counter()

    for baris in proses.stdout:
        try:
            ev = json.loads(baris)
        except json.JSONDecodeError:
            continue

        if ev["t"] == "galat":
            proses.wait()
            raise gr.Error(f"{cfg['skrip'].name}: {ev['pesan']}")

        if ev["t"] == "mulai":
            yield (
                gr.skip(),
                "",
                "",
                f"{ev['mesin']} siap ({ev['detik_muat']} s) — {cfg['catatan']}",
                None,
            )

        elif ev["t"] == "halaman_mulai":
            n = ev["n"]
            # Engine batch membaca satu halaman utuh dulu, baru mengirim semua bloknya.
            # Halamannya ditampilkan sekarang supaya jelas apa yang sedang dikerjakan.
            yield (
                pil_per_halaman[n],
                gr.skip(),
                gr.skip(),
                f"Membaca halaman {n}/{len(gambar)} — engine ini memproses satu halaman "
                f"utuh sebelum hasilnya keluar ({time.perf_counter() - t_awal:.0f} s berjalan)",
                n,
            )

        elif ev["t"] == "blok":
            n = ev["n"]
            blok = blok_per_halaman.setdefault(n, [])
            blok.append((ev["kategori"], tuple(ev["bbox"]), ev["teks"]))
            pil = pil_per_halaman[n]
            buat_gambar = pembuat_gambar(pil, cache)

            time.sleep(JEDA_BLOK)
            yield (
                gambar_kotak(pil, [b for _, b, _ in blok]),
                "\n".join(selesai + [f"<!-- halaman {n} -->", *baris_mentah(blok)]),
                "\n\n".join(
                    selesai_md
                    + [f"<!-- halaman {n} -->\n\n{blok_ke_markdown(blok, buat_gambar)}"]
                ),
                f"Halaman {n}/{len(gambar)} — {len(blok)} blok, "
                f"{time.perf_counter() - t_awal:.1f} s",
                n,
            )

        elif ev["t"] == "halaman_selesai":
            n = ev["n"]
            blok = blok_per_halaman.get(n, [])
            buat_gambar = pembuat_gambar(pil_per_halaman[n], cache)
            selesai += [f"<!-- halaman {n} -->", *baris_mentah(blok)]
            selesai_md.append(
                f"<!-- halaman {n} -->\n\n{blok_ke_markdown(blok, buat_gambar)}"
            )

    proses.wait()
    yield _akhir(
        selesai, selesai_md, len(gambar), time.perf_counter() - t_awal, pisah_mentah="\n"
    )


def _akhir(selesai, selesai_md, jumlah, total, pisah_mentah="\n\n"):
    return (
        gr.skip(),
        pisah_mentah.join(selesai),
        "\n\n".join(selesai_md),
        f"**Selesai** — {jumlah} halaman dalam {total:.1f} s ({total / jumlah:.1f} s/halaman)",
    )


def jalankan(berkas, nama_mesin, dpi, max_tokens, prompt):
    if not berkas:
        raise gr.Error("Pilih file PDF atau gambar dulu.")

    cfg = MESIN[nama_mesin]
    src = Path(berkas)

    with tempfile.TemporaryDirectory() as tmpdir:
        gambar = siapkan_halaman(src, dpi, tmpdir)
        halaman = [(n, Image.open(p).convert("RGB")) for n, p in gambar]
        cache = {}
        # Gambar terakhir tiap halaman — mula-mula halaman polos, lalu diganti versi
        # ber-highlight. Dipakai tombol halaman setelah (dan selama) OCR berjalan.
        daftar = [pil for _, pil in halaman]
        kini = 0

        def label(i):
            return f"Halaman {i + 1} / {len(daftar)}"

        yield (daftar[0], "", "", f"{nama_mesin} — {len(gambar)} halaman, menyiapkan...",
               daftar, kini, label(kini))

        if cfg["jenis"] == "lokal":
            sumber = jalankan_unlimited(gambar, halaman, max_tokens, prompt, cache)
        else:
            if cfg.get("server"):
                pastikan_server(cfg, nama_mesin)
            if cfg["umpan"] == "dokumen" and src.suffix.lower() != ".pdf":
                raise gr.Error(f"{nama_mesin} pada demo ini hanya menerima PDF.")
            sumber = jalankan_subproses(cfg, src, gambar, halaman, cache)

        for keluaran in sumber:
            img, mentah_, md, status_, n = (*keluaran, None)[:5]
            if n is not None:
                kini = n - 1
                if isinstance(img, Image.Image):
                    daftar[kini] = img
            yield img, mentah_, md, status_, daftar, kini, label(kini)


# Tinggi ketiga panel dikunci dan digulir sendiri-sendiri. Tanpa ini, panel tengah
# tumbuh mengikuti isi lalu mendorong tata letak tiap kali stream memperbarui.
CSS = """
.kolom-panel { height: 640px; overflow: hidden; }
.kolom-panel .gradio-image, .kolom-panel img { object-fit: contain; }
.kolom-panel .prose { overflow-y: auto; height: 100%; }
/* Tinggi textarea dipaksa: Gradio menumbuhkannya mengikuti isi, jadi panel tengah
   melompat sekali saat teks pertama masuk. */
.kolom-panel textarea { height: 560px !important; resize: none; }
/* Kotak unggah bawaan setinggi ~320px mendorong ketiga panel keluar layar. */
#kotak-unggah { height: 130px; }
.label-halaman { text-align: center; align-self: center; min-width: 120px; }
/* Tanpa ini, teks Halaman 1 / 1 terpotong jadi dua baris di antara kedua tombol. */
.label-halaman * { white-space: nowrap; }
#kotak-unggah .wrap { min-height: 0; }
"""

def geser_halaman(daftar, indeks, arah):
    if not daftar:
        return gr.skip(), indeks, gr.skip()
    i = max(0, min(len(daftar) - 1, indeks + arah))
    return daftar[i], i, f"Halaman {i + 1} / {len(daftar)}"


def _sel_nilai(isi):
    if not isi:
        return "galat"
    if isi["status"] == "tidak ditemukan":
        return "— (tidak ditemukan)"
    return f"{isi['nilai']}  [{isi['status']}]"


KOLOM_BIAYA = [
    "Model", "Tingkat", "Token masuk", "Token keluar", "— berpikir",
    "Biaya/dokumen (USD)", "Per 1.000 dokumen (USD)", "Waktu (s)",
    "Field ditemukan", "Terbukti",
]


def jalankan_ekstraksi(markdown, field, model_dipilih, berkas, nama_mesin, daftar_halaman):
    """Semua model terpilih mengambil field dari teks OCR yang SAMA, dijalankan paralel.

    Teks yang sama untuk semua model, jadi beda token dan biaya di tabel murni beda
    model — bukan beda hasil OCR. Tabel diperbarui setiap satu model selesai.
    """
    field = [f.strip() for f in (field or []) if f and f.strip()]
    if not (markdown or "").strip():
        raise gr.Error("Jalankan OCR dulu.")
    if not field:
        raise gr.Error("Isi minimal satu field.")
    if not model_dipilih:
        raise gr.Error("Pilih minimal satu model.")

    teks = teks_untuk_llm(markdown)
    dokumen = Path(berkas).name if berkas else "-"
    nama = {i: n for i, n, *_ in MODEL_LLM}
    tingkat = {i: t for i, _, t, *_ in MODEL_LLM}
    baris_biaya, hasil_per_model, detail = [], {}, {}
    # Diserahkan ke langkah analisis: field tiap model dan biaya ekstraksinya.
    serah = {}

    def tabel():
        # Urut termurah dulu; model yang galat di paling bawah.
        biaya = pd.DataFrame(
            sorted(baris_biaya, key=lambda b: (b[5] == "", b[5] if b[5] != "" else 0)),
            columns=KOLOM_BIAYA,
        )
        urutan = [m for m in model_dipilih if m in hasil_per_model]
        nilai = pd.DataFrame(
            [[f] + [_sel_nilai((hasil_per_model[m] or {}).get(f)) for m in urutan] for f in field],
            columns=["Field"] + [nama.get(m, m) for m in urutan],
        )
        return biaya, nilai

    yield (gr.skip(), gr.skip(),
           f"Mengirim {len(teks):,} karakter teks OCR ke {len(model_dipilih)} model sekaligus...",
           gr.skip(), gr.skip())

    with ThreadPoolExecutor(max_workers=len(model_dipilih)) as pool:
        tugas = {pool.submit(ambil_field, teks, field, m): m for m in model_dipilih}
        for selesai in as_completed(tugas):
            m = tugas[selesai]
            try:
                hasil, p, mentah = selesai.result()
            except Exception as e:
                hasil_per_model[m] = None
                detail[m] = {"galat": str(e)}
                baris_biaya.append([nama.get(m, m), tingkat.get(m, ""), "", "", "", "", "", "",
                                    "galat", str(e)[:80]])
            else:
                semua = (hasil or {}).values()
                ditemukan = sum(1 for h in semua if h["status"] != "tidak ditemukan")
                terbukti = sum(1 for h in (hasil or {}).values() if h["status"] == "terbukti")
                # Dicatat di utas utama, bukan di utas pekerja: menulis Excel dari beberapa
                # utas sekaligus bisa menimpa baris satu sama lain.
                catat(
                    tahap="ekstraksi",
                    dokumen=dokumen, mesin_ocr=nama_mesin, halaman=len(daftar_halaman or []),
                    karakter_dikirim=len(teks), model=m, field_diminta=len(field),
                    field_ditemukan=ditemukan, field_terbukti=terbukti, pemakaian=p,
                    status="ok" if hasil is not None else "json tidak terbaca",
                )
                hasil_per_model[m] = hasil
                if hasil is not None:
                    serah[m] = {"hasil": hasil, "biaya_usd": p["biaya_usd"]}
                detail[m] = {"pemakaian": p, "hasil": hasil}
                if hasil is None:
                    detail[m]["jawaban_mentah"] = mentah
                baris_biaya.append([
                    nama.get(m, m), tingkat.get(m, ""), p["token_masuk"], p["token_keluar"],
                    p["token_berpikir"], round(p["biaya_usd"], 6), round(p["biaya_usd"] * 1000, 2),
                    p["detik"], f"{ditemukan}/{len(field)}", f"{terbukti}/{len(field)}",
                ])
            biaya, nilai = tabel()
            yield (biaya, nilai,
                   f"{len(hasil_per_model)}/{len(model_dipilih)} model selesai · "
                   f"teks OCR {len(teks):,} karakter",
                   detail, serah)

    biaya, nilai = tabel()
    yield (biaya, nilai,
           f"**Selesai** — {len(model_dipilih)} model · teks OCR {len(teks):,} karakter · "
           f"tercatat di `hasil/biaya-token-llm.xlsx`",
           detail, serah)


def baca_file(path):
    """Isi file unggahan (.json / .md) untuk dimasukkan ke editornya."""
    return Path(path).read_text(encoding="utf-8") if path else gr.skip()


def isi_scan_md(markdown):
    """Setelah OCR: salin teksnya — tanpa potongan gambar — ke masukan analisis."""
    return teks_untuk_llm(markdown)


KOLOM_BIAYA_ANALISIS = [
    "Model", "Tingkat", "Token masuk", "— dari cache", "Token keluar", "— berpikir",
    "Biaya analisis (USD)", "Biaya ekstraksi (USD)", "Total per dokumen (USD)",
    "Per 1.000 dokumen (USD)", "Waktu (s)",
]


def jalankan_analisis(json_hazard, scan_md, system_prompt, model_dipilih, serah,
                      berkas, nama_mesin, daftar_halaman):
    """Semua model terpilih menganalisis tiga masukan yang SAMA, dijalankan paralel.

    Masukan yang sama untuk semua model, jadi beda token dan biaya murni beda model.
    """
    json_hazard = (json_hazard or "").strip()
    if json_hazard:
        try:
            json.loads(json_hazard)
        except json.JSONDecodeError as e:
            raise gr.Error(f"JSON hazard tidak sah: {e}")
    scan_md = (scan_md or "").strip()
    if not json_hazard and not scan_md:
        raise gr.Error("JSON hazard dan markdown scan OCR dua-duanya kosong — tidak ada yang "
                       "bisa dianalisis.")
    if not (system_prompt or "").strip():
        raise gr.Error("System prompt kosong.")
    if not model_dipilih:
        raise gr.Error("Pilih minimal satu model.")

    ukuran = {"prompt": len(system_prompt), "json": len(json_hazard), "md": len(scan_md)}
    rincian = (f"system prompt {ukuran['prompt']:,} · JSON hazard {ukuran['json']:,} · "
               f"scan OCR {ukuran['md']:,} karakter")
    if not json_hazard:
        rincian += " · **tanpa JSON hazard**"
    if not scan_md:
        rincian += " · **tanpa scan OCR**"
    dokumen = Path(berkas).name if berkas else "-"
    nama = {i: n for i, n, *_ in MODEL_LLM}
    tingkat = {i: t for i, _, t, *_ in MODEL_LLM}
    baris_biaya, jawaban, detail = [], {}, {}

    def tampilan():
        # Urut termurah dulu (total bila ada, selain itu biaya analisis); galat di bawah.
        biaya = pd.DataFrame(
            sorted(baris_biaya, key=lambda b: (b[6] == "", b[8] == "", b[8] or b[6] or 0)),
            columns=KOLOM_BIAYA_ANALISIS,
        )
        teks = "\n\n---\n\n".join(
            f"### {nama.get(m, m)}\n\n{jawaban[m]}" for m in model_dipilih if m in jawaban
        )
        return biaya, teks

    yield (gr.skip(), gr.skip(), f"Menganalisis dengan {len(model_dipilih)} model · {rincian}",
           gr.skip())

    with ThreadPoolExecutor(max_workers=len(model_dipilih)) as pool:
        tugas = {pool.submit(analisis, json_hazard, scan_md, system_prompt, m): m
                 for m in model_dipilih}
        for selesai in as_completed(tugas):
            m = tugas[selesai]
            try:
                teks, p = selesai.result()
            except Exception as e:
                detail[m] = {"galat": str(e)}
                jawaban[m] = f"_Galat: {e}_"
                baris_biaya.append([nama.get(m, m), tingkat.get(m, "")] + [""] * 9)
            else:
                # Dicatat di utas utama: menulis Excel dari beberapa utas bisa saling timpa.
                catat(
                    tahap="analisis", dokumen=dokumen, mesin_ocr=nama_mesin,
                    halaman=len(daftar_halaman or []), karakter_dikirim=sum(ukuran.values()),
                    karakter_system_prompt=ukuran["prompt"], karakter_json_hazard=ukuran["json"],
                    karakter_scan_md=ukuran["md"], model=m, pemakaian=p,
                    status="ok" if teks.strip() else "jawaban kosong",
                )
                biaya_ekstraksi = (serah or {}).get(m, {}).get("biaya_usd")
                total = p["biaya_usd"] + biaya_ekstraksi if biaya_ekstraksi is not None else None
                baris_biaya.append([
                    nama.get(m, m), tingkat.get(m, ""), p["token_masuk"], p["token_cache"],
                    p["token_keluar"], p["token_berpikir"], round(p["biaya_usd"], 6),
                    round(biaya_ekstraksi, 6) if biaya_ekstraksi is not None else "",
                    round(total, 6) if total is not None else "",
                    round(total * 1000, 2) if total is not None else "",
                    p["detik"],
                ])
                jawaban[m] = teks.strip() or "_Jawaban kosong._"
                detail[m] = {"pemakaian": p}
            biaya, teks_md = tampilan()
            yield (biaya, teks_md, f"{len(jawaban)}/{len(model_dipilih)} model selesai · {rincian}",
                   detail)

    biaya, teks_md = tampilan()
    yield (biaya, teks_md,
           f"**Selesai** — {len(model_dipilih)} model · {rincian} · "
           f"tercatat di `hasil/biaya-token-llm.xlsx`",
           detail)


def llm_aktif():
    """Sakelar OPEN_ROUTER_ENALBLE dibaca ulang dari .env setiap kali, bukan sekali di awal:
    mengubah .env lalu me-refresh browser sudah cukup, tanpa restart app."""
    return baca_env().get("OPEN_ROUTER_ENALBLE", "true").lower() == "true"


def atur_tampilan():
    """Saat halaman dimuat: tampilkan bagian LLM ATAU bagian parsing aturan, sesuai sakelar."""
    aktif = llm_aktif()
    return gr.Column(visible=aktif), gr.Column(visible=not aktif)


KOLOM_FIELD_ATURAN = ["Field", "Nilai", "Label di dokumen / alasan", "Cara", "Sumber",
                      "Halaman", "Catatan"]


def isi_field_aturan(mentah_ocr, diminta, berkas, dpi):
    """Setelah OCR: nilai field lewat parser tanpa LLM (tanpa token).

    Parser membaca output MENTAH, bukan markdown, karena butuh posisi (bbox) tiap blok:
    label dan nilainya dipasangkan juga lewat letak, bukan hanya lewat tabel.

    Halaman dirender ulang dan barisnya dideteksi (±1,4 detik per halaman) untuk
    menemukan kotak isian formulir. Tanpa itu, formulir dua kolom — label di kiri, kotak
    jawaban di kanan — tidak bisa dipasangkan.
    """
    diminta = [f.strip() for f in (diminta or []) if f and f.strip()]
    if not diminta or not (mentah_ocr or "").strip():
        return pd.DataFrame(columns=KOLOM_FIELD_ATURAN)
    hasil = cari_field(mentah_ocr, diminta, tata_letak(berkas, int(dpi or 200)))
    baris = []
    for f in diminta:
        h = hasil.get(f, {})
        if h.get("nilai"):
            hal = h.get("halaman")
            baris.append([f, h["nilai"], h.get("label_ditemukan", ""), h.get("cara", ""),
                          h.get("sumber", ""), "" if hal is None else str(hal),
                          h.get("catatan", "")])
        else:
            label = h.get("label_ditemukan")
            alasan = h.get("alasan", "tidak ditemukan") + (f" ({label})" if label else "")
            baris.append([f, "", alasan, "", "", "", ""])
    return pd.DataFrame(baris, columns=KOLOM_FIELD_ATURAN)


def _label_model(m):
    id_model, nama, tingkat, skor, masuk, keluar = m
    return (f"{nama} · {tingkat} · skor {skor} · ${masuk:g}/${keluar:g} per 1 jt token", id_model)


with gr.Blocks(title="Banding OCR — Apple Silicon") as demo:
    gr.Markdown(
        "# Banding OCR di Apple Silicon\n"
        "Tiga engine, satu tampilan. Highlight mengikuti blok yang sedang dibaca."
    )

    with gr.Row():
        berkas = gr.File(
            label="PDF atau gambar",
            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"],
            type="filepath",
            scale=3,
            elem_id="kotak-unggah",
        )
        mesin = gr.Dropdown(list(MESIN), value=list(MESIN)[0], label="Engine", scale=2)
    with gr.Row():
        dpi = gr.Slider(100, 300, value=200, step=50, label="DPI render PDF")
        max_tokens = gr.Slider(
            512, 8192, value=4096, step=512, label="Maks token (Unlimited-OCR saja)"
        )
    prompt = gr.Textbox(value=PROMPT, label="Prompt (Unlimited-OCR saja)")
    field_pilih = gr.Dropdown(
        FIELD_BAWAAN,
        value=FIELD_BAWAAN,
        multiselect=True,
        allow_custom_value=True,
        label="Field yang dicari",
        info="Tekan Enter untuk menambah. Tulis bahasa biasa: dengan LLM artinya "
             "dicocokkan, tanpa LLM parser memakai kamus sebutan dan tahan salah baca "
             "OCR ringan. Untuk label tak lazim, tulis mirip yang tertulis di dokumen.",
    )
    tombol = gr.Button("Jalankan OCR", variant="primary")
    status = gr.Markdown("")

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes=["kolom-panel"]):
            gr.Markdown("### Dokumen")
            tampil = gr.Image(show_label=False, height=530, type="pil")
            with gr.Row():
                tombol_sebelum = gr.Button("◀ Sebelumnya", size="sm")
                label_halaman = gr.Markdown("", elem_classes=["label-halaman"])
                tombol_sesudah = gr.Button("Berikutnya ▶", size="sm")
        with gr.Column(elem_classes=["kolom-panel"]):
            gr.Markdown("### Output mentah")
            # Textbox, bukan Code: hanya Textbox yang punya autoscroll. Tanpa itu,
            # gulirannya melompat ke atas tiap pembaruan karena isinya diganti utuh.
            mentah = gr.Textbox(
                show_label=False,
                lines=24,
                max_lines=24,
                autoscroll=True,
            )
        with gr.Column(elem_classes=["kolom-panel"]):
            gr.Markdown("### Markdown")
            bersih = gr.Markdown(height=580, container=True)

    gambar_halaman = gr.State([])
    indeks_halaman = gr.State(0)

    with gr.Column(visible=not llm_aktif()) as bagian_aturan:
        gr.Markdown(
            "### Nilai field — parsing aturan\n"
            "LLM dimatikan (`OPEN_ROUTER_ENALBLE=false` di `.env`), jadi nilai field dicari "
            "parser tanpa token ([`parser_field.py`](parser_field.py)). Label dicocokkan "
            "dengan kamus sebutan (\"nama lengkap\" juga menemukan \"Nama Tertanggung\"), "
            "lalu nilainya diambil dari baris yang sama, kotak tabel sebelahnya, **kotak "
            "isian formulir** yang sejajar dengan label, atau letaknya (kanan/bawah label). "
            "Batas kotak isian dibaca dari garis cetak formulir, jadi kotak yang memang "
            "tidak diisi tetap kosong. Kalau nilainya meragukan, kolom dibiarkan kosong "
            "beserta alasannya — bukan ditebak."
        )
        tabel_field_aturan = gr.Dataframe(headers=KOLOM_FIELD_ATURAN, interactive=False,
                                          wrap=True)

    with gr.Column(visible=llm_aktif()) as bagian_llm:
        gr.Markdown(
            "### Ambil field dengan LLM\n"
            "Semua model membaca teks OCR yang sama, jadi beda token dan biaya di bawah murni "
            "beda model. Skor kualitas: BenchLM · harga: OpenRouter, USD per 1 juta token "
            "masuk/keluar."
        )
        with gr.Row():
            model_llm = gr.Dropdown(
                [_label_model(m) for m in MODEL_LLM],
                value=MODEL_BAWAAN,
                multiselect=True,
                label="Model yang dibandingkan",
                scale=4,
            )
            tombol_llm = gr.Button("Ambil field dengan LLM", variant="primary", scale=1)
        status_llm = gr.Markdown("")
        gr.Markdown("#### Token dan biaya per dokumen")
        tabel_biaya = gr.Dataframe(interactive=False, wrap=True)
        gr.Markdown(
            "#### Nilai field per model\n"
            "`terbukti` = tertulis di teks OCR · `ditafsirkan` = LLM membetulkan salah baca OCR, "
            "belum tentu benar · `tidak terbukti` = kutipannya tidak ada di teks OCR, jangan dipercaya"
        )
        tabel_nilai = gr.Dataframe(interactive=False, wrap=True)
        with gr.Accordion("Rincian jawaban per model (JSON)", open=False):
            detail_llm = gr.JSON()
        hasil_ekstraksi = gr.State({})

        gr.Markdown(
            "### Analisis risiko LLM\n"
            "Tiga masukan: **JSON hazard** (sementara simulasi — nantinya dari API risk analysis), "
            "**markdown scan OCR** (terisi otomatis setelah OCR), dan **system prompt**. System "
            "prompt dikirim paling depan dan sama persis di setiap dokumen — bagian itulah yang "
            "bisa di-cache penyedia model."
        )
        with gr.Row():
            with gr.Column():
                file_json = gr.File(label="Unggah JSON hazard (.json)", file_types=[".json"],
                                    type="filepath")
                json_hazard = gr.Code(
                    value=(AKAR / "contoh" / "hazard-api-contoh.json").read_text(encoding="utf-8"),
                    language="json",
                    label="JSON hazard — terisi contoh dari API; ganti, tempel, atau unggah",
                    lines=14,
                )
            with gr.Column():
                file_prompt = gr.File(label="Unggah system prompt (.md)", file_types=[".md", ".txt"],
                                      type="filepath")
                prompt_analisis = gr.Textbox(value=PROMPT_ANALISIS, label="System prompt", lines=14)
        with gr.Accordion("Markdown scan OCR yang dikirim", open=False):
            scan_md = gr.Code(language="markdown", lines=16,
                              label="Terisi otomatis setelah OCR; boleh diedit atau ditempel")
        with gr.Row():
            model_analisis = gr.Dropdown(
                [_label_model(m) for m in MODEL_LLM],
                value=MODEL_BAWAAN,
                multiselect=True,
                label="Model analisis",
                scale=4,
            )
            tombol_analisis = gr.Button("Analisis", variant="primary", scale=1)
        status_analisis = gr.Markdown("")
        gr.Markdown(
            "#### Token dan biaya per dokumen — ekstraksi + analisis\n"
            "Biaya ekstraksi diambil dari model yang sama pada langkah di atas. Total hanya terisi "
            "kalau model itu menjalani kedua langkah."
        )
        tabel_biaya_analisis = gr.Dataframe(interactive=False, wrap=True)
        gr.Markdown("#### Hasil analisis per model")
        hasil_analisis = gr.Markdown("")
        with gr.Accordion("Rincian pemakaian per model (JSON)", open=False):
            detail_analisis = gr.JSON()

    tombol.click(
        jalankan,
        inputs=[berkas, mesin, dpi, max_tokens, prompt],
        outputs=[tampil, mentah, bersih, status, gambar_halaman, indeks_halaman, label_halaman],
    ).success(isi_scan_md, inputs=[bersih], outputs=[scan_md]).success(
        isi_field_aturan, inputs=[mentah, field_pilih, berkas, dpi],
        outputs=[tabel_field_aturan])

    demo.load(atur_tampilan, outputs=[bagian_llm, bagian_aturan])

    tombol_sebelum.click(
        lambda d, i: geser_halaman(d, i, -1),
        inputs=[gambar_halaman, indeks_halaman],
        outputs=[tampil, indeks_halaman, label_halaman],
    )
    tombol_sesudah.click(
        lambda d, i: geser_halaman(d, i, 1),
        inputs=[gambar_halaman, indeks_halaman],
        outputs=[tampil, indeks_halaman, label_halaman],
    )

    tombol_llm.click(
        jalankan_ekstraksi,
        inputs=[bersih, field_pilih, model_llm, berkas, mesin, gambar_halaman],
        outputs=[tabel_biaya, tabel_nilai, status_llm, detail_llm, hasil_ekstraksi],
    )

    file_json.upload(baca_file, inputs=[file_json], outputs=[json_hazard])
    file_prompt.upload(baca_file, inputs=[file_prompt], outputs=[prompt_analisis])

    tombol_analisis.click(
        jalankan_analisis,
        inputs=[json_hazard, scan_md, prompt_analisis, model_analisis, hasil_ekstraksi,
                berkas, mesin, gambar_halaman],
        outputs=[tabel_biaya_analisis, hasil_analisis, status_analisis, detail_analisis],
    )

if __name__ == "__main__":
    # css pindah ke launch() sejak Gradio 6.
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, css=CSS)
