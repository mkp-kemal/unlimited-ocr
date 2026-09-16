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
import subprocess
import tempfile
import time
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

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
        yield halaman[0][1], "", "", f"{nama_mesin} — {len(gambar)} halaman, menyiapkan..."

        if cfg["jenis"] == "lokal":
            yield from jalankan_unlimited(gambar, halaman, max_tokens, prompt, cache)
        else:
            if cfg.get("server"):
                pastikan_server(cfg, nama_mesin)
            if cfg["umpan"] == "dokumen" and src.suffix.lower() != ".pdf":
                raise gr.Error(f"{nama_mesin} pada demo ini hanya menerima PDF.")
            yield from jalankan_subproses(cfg, src, gambar, halaman, cache)


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
        )
        mesin = gr.Dropdown(list(MESIN), value=list(MESIN)[0], label="Engine", scale=2)
    with gr.Row():
        dpi = gr.Slider(100, 300, value=200, step=50, label="DPI render PDF")
        max_tokens = gr.Slider(
            512, 8192, value=4096, step=512, label="Maks token (Unlimited-OCR saja)"
        )
    prompt = gr.Textbox(value=PROMPT, label="Prompt (Unlimited-OCR saja)")
    tombol = gr.Button("Jalankan OCR", variant="primary")
    status = gr.Markdown("")

    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown("### Dokumen")
            tampil = gr.Image(show_label=False, height=620, type="pil")
        with gr.Column():
            gr.Markdown("### Output mentah")
            mentah = gr.Code(lines=28, wrap_lines=True)
        with gr.Column():
            gr.Markdown("### Markdown")
            bersih = gr.Markdown(height=620, container=True)

    tombol.click(
        jalankan,
        inputs=[berkas, mesin, dpi, max_tokens, prompt],
        outputs=[tampil, mentah, bersih, status],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
