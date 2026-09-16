#!/usr/bin/env python
"""OCR satu file (gambar atau PDF) dengan Unlimited-OCR lewat MLX di Apple Silicon.

Cetak hasil + waktu per halaman, simpan ke folder keluaran.
"""

import argparse
import itertools
import json
import re
import sys
import tempfile
import time
from pathlib import Path

MODEL = "mlx-community/Unlimited-OCR-8bit"
PROMPT = "document parsing."
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def pdf_ke_gambar(pdf_path, dpi, halaman, tmpdir):
    import fitz

    doc = fitz.open(pdf_path)
    if halaman:
        awal, akhir = halaman
        indeks = range(awal - 1, min(akhir, doc.page_count))
    else:
        indeks = range(doc.page_count)

    berkas = []
    for i in indeks:
        pix = doc[i].get_pixmap(dpi=dpi)
        out = Path(tmpdir) / f"hal-{i + 1:03d}.png"
        pix.save(out)
        berkas.append((i + 1, out))
    doc.close()
    return berkas


def perbesar_bila_kecil(img_path, tmpdir, target_sisi_panjang=1800):
    """Gambar kecil kehilangan banyak teks saat di-OCR.

    Patokan sisi panjang, bukan sisi pendek: pada uji kwitansi 600x227, menaikkan
    sisi pendek ke 1024 (4.5x) justru membuat model melewatkan blok judul yang
    tertangkap pada 3x. Angka 1800 disetel dari satu sampel itu.
    """
    from PIL import Image

    im = Image.open(img_path)
    sisi_panjang = max(im.size)
    if sisi_panjang >= target_sisi_panjang:
        return img_path

    skala = target_sisi_panjang / sisi_panjang
    out = Path(tmpdir) / f"{img_path.stem}-besar.png"
    im.resize((round(im.width * skala), round(im.height * skala)), Image.LANCZOS).save(out)
    print(
        f"Gambar diperbesar {skala:.1f}x: {im.size} -> "
        f"{(round(im.width * skala), round(im.height * skala))}",
        file=sys.stderr,
    )
    return out


# Model mengeluarkan bbox pada skala 0-999, bukan piksel.
SKALA = 1000.0
DET = re.compile(r"<\|det\|>([a-z_]+)\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]<\|/det\|>")


def blok_det(teks):
    """Pecah keluaran model jadi teks pembuka + daftar (kategori, bbox, isi)."""
    cocok = list(DET.finditer(teks))
    if not cocok:
        return teks.strip(), []

    blok = []
    for i, m in enumerate(cocok):
        batas = cocok[i + 1].start() if i + 1 < len(cocok) else len(teks)
        blok.append(
            (m.group(1), tuple(int(g) for g in m.groups()[1:]), teks[m.end() : batas].strip())
        )
    return teks[: cocok[0].start()].strip(), blok


def potong_bbox(pil, bbox):
    """Ambil potongan halaman sesuai bbox model (skala 0-999)."""
    x1, y1, x2, y2 = bbox
    w, h = pil.size
    return pil.crop(
        (
            round(x1 / SKALA * w),
            round(y1 / SKALA * h),
            round(x2 / SKALA * w),
            round(y2 / SKALA * h),
        )
    )


# Kategori blok yang isinya gambar, bukan teks. Model memakai nama berbeda-beda —
# `chart` untuk grafik, `image` untuk foto/logo — dan semuanya keluar tanpa teks.
VISUAL = ("image", "figure", "chart", "diagram", "graph", "seal", "stamp", "logo")


def blok_ke_markdown(blok, pada_gambar=None):
    """Susun Markdown dari daftar (kategori, bbox, isi) mana pun engine-nya.

    Nama kategori berbeda antar engine — Unlimited-OCR memakai `title`, PaddleOCR-VL
    `paragraph_title`, MinerU `title` — jadi dicocokkan longgar, bukan disamakan persis.
    """
    baris = []
    for kategori, bbox, isi in blok:
        k = (kategori or "").lower()
        # "image_caption" mengandung "image" tapi isinya teks keterangan, bukan gambar.
        gambar = "caption" not in k and (any(v in k for v in VISUAL) or not isi)
        if gambar:
            md = pada_gambar(bbox) if pada_gambar else None
            if md:
                baris.append(md)
            continue
        if not isi:
            continue
        if "title" in k:
            baris.append(f"## {isi}")
        elif "page_number" in k:
            baris.append(f"*{isi}*")
        else:
            baris.append(isi)
    return "\n\n".join(baris)


def ke_markdown(teks, pada_gambar=None):
    """Ubah keluaran mentah Unlimited-OCR jadi Markdown wajar.

    `pada_gambar(bbox)` dipanggil untuk tiap blok gambar dan boleh mengembalikan
    sintaks gambar Markdown; kalau None, blok gambar dilewati.
    """
    pembuka, blok = blok_det(teks)
    isi = blok_ke_markdown(blok, pada_gambar)
    return f"{pembuka}\n\n{isi}" if pembuka else isi


def simpan_potongan(img_path, nomor, dir_gambar):
    """Kembalikan callback yang menyimpan blok gambar ke disk lalu merujuknya dari Markdown."""
    from PIL import Image

    pil = Image.open(img_path).convert("RGB")
    urut = itertools.count(1)

    def simpan(bbox):
        potong = potong_bbox(pil, bbox)
        if min(potong.size) < 8:  # sisa deteksi yang terlalu tipis untuk berguna
            return None
        dir_gambar.mkdir(parents=True, exist_ok=True)
        nama = f"hal{nomor}-{next(urut):02d}.png"
        potong.save(dir_gambar / nama)
        return f"![blok gambar halaman {nomor}]({dir_gambar.name}/{nama})"

    return simpan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("berkas", help="file gambar atau PDF")
    p.add_argument("--out", default="hasil", help="folder keluaran (default: hasil)")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--dpi", type=int, default=200, help="resolusi render PDF")
    p.add_argument("--halaman", help="rentang halaman PDF, mis. 1-5")
    args = p.parse_args()

    src = Path(args.berkas)
    if not src.exists():
        sys.exit(f"File tidak ditemukan: {src}")

    halaman = None
    if args.halaman:
        awal, _, akhir = args.halaman.partition("-")
        halaman = (int(awal), int(akhir or awal))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dir_gambar = out_dir / f"{src.stem}-gambar"

    t0 = time.perf_counter()
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model, processor = load(args.model)
    config = load_config(args.model)
    t_muat = time.perf_counter() - t0
    print(f"Model dimuat dalam {t_muat:.1f} s ({args.model})", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmpdir:
        if src.suffix.lower() == ".pdf":
            t = time.perf_counter()
            gambar = pdf_ke_gambar(src, args.dpi, halaman, tmpdir)
            print(
                f"PDF dirender: {len(gambar)} halaman @ {args.dpi} dpi "
                f"({time.perf_counter() - t:.1f} s)",
                file=sys.stderr,
            )
        elif src.suffix.lower() in IMAGE_EXT:
            gambar = [(1, perbesar_bila_kecil(src, tmpdir))]
        else:
            sys.exit(f"Format tidak didukung: {src.suffix}")

        prompt = apply_chat_template(processor, config, args.prompt, num_images=1)

        potongan, potongan_bersih, catatan = [], [], []
        t_mulai = time.perf_counter()
        for nomor, img in gambar:
            t = time.perf_counter()
            hasil = generate(
                model,
                processor,
                prompt,
                [str(img)],
                max_tokens=args.max_tokens,
                temperature=0.0,
                verbose=False,
            )
            durasi = time.perf_counter() - t
            teks = hasil.text if hasattr(hasil, "text") else str(hasil)
            potongan.append(f"<!-- halaman {nomor} -->\n{teks.strip()}")
            potongan_bersih.append(
                f"<!-- halaman {nomor} -->\n\n"
                + ke_markdown(teks, simpan_potongan(img, nomor, dir_gambar))
            )
            catatan.append(
                {
                    "halaman": nomor,
                    "detik": round(durasi, 2),
                    "token_keluar": getattr(hasil, "generation_tokens", None),
                    "token_per_detik": round(getattr(hasil, "generation_tps", 0) or 0, 1),
                    "karakter": len(teks),
                }
            )
            print(
                f"  halaman {nomor}/{len(gambar)}: {durasi:.1f} s, "
                f"{catatan[-1]['token_keluar']} token, {len(teks)} karakter",
                file=sys.stderr,
            )
        t_total = time.perf_counter() - t_mulai

    teks_md = "\n\n".join(potongan)
    (out_dir / f"{src.stem}.md").write_text(teks_md, encoding="utf-8")
    (out_dir / f"{src.stem}.bersih.md").write_text(
        "\n\n".join(potongan_bersih), encoding="utf-8"
    )
    (out_dir / f"{src.stem}.json").write_text(
        json.dumps(
            {
                "berkas": str(src),
                "model": args.model,
                "prompt": args.prompt,
                "dpi": args.dpi if src.suffix.lower() == ".pdf" else None,
                "detik_muat_model": round(t_muat, 2),
                "detik_ocr_total": round(t_total, 2),
                "halaman": catatan,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSelesai: {len(catatan)} halaman dalam {t_total:.1f} s "
        f"({t_total / len(catatan):.1f} s/halaman)\n"
        f"Hasil: {out_dir / (src.stem + '.md')} (mentah)\n"
        f"       {out_dir / (src.stem + '.bersih.md')} (Markdown)",
        file=sys.stderr,
    )
    print(teks_md)


if __name__ == "__main__":
    main()
