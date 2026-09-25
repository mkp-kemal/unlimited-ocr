#!/usr/bin/env python
"""Runner MinerU: dokumen -> baris JSON ke stdout, satu halaman sekali jalan.

Dijalankan sebagai subprocess dengan ../.venv-mineru/bin/python. Kontrak keluarannya
sama dengan runner lain.

Halaman diproses satu per satu lewat do_parse() supaya blok bisa dikirim begitu satu
halaman rampung, bukan menunggu seluruh dokumen. Model tetap dimuat sekali karena
seluruh loop berjalan dalam satu proses.

Device: MPS. Jalur MLX milik MinerU tidak dipakai karena mlx-vlm menuntut
transformers 5.x sedangkan MinerU mensyaratkan <5.0.0.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MINERU_DEVICE_MODE", "mps")
os.environ.setdefault("MINERU_MODEL_SOURCE", "huggingface")

SKALA = 1000.0

# "pipeline" cepat (±9 detik/halaman) tapi struktur tabelnya sering rusak pada formulir
# tulisan tangan: label dan nilai field lain bisa tergabung dalam satu sel, sehingga
# "Nama Lengkap" tidak punya nilai. "hybrid-engine" memakai VLM untuk isi tabel dan
# memberi sel yang benar, dengan ongkos ±61 detik/halaman — terukur pada SPPA JASINDO.
BACKEND = os.environ.get("MINERU_BACKEND", "pipeline")


def kirim(**ev):
    print(json.dumps(ev, ensure_ascii=False), flush=True)


def isi_blok(blok):
    """Gabungkan span jadi satu teks, termasuk dari sub-blok.

    Tabel menyimpan isinya satu tingkat lebih dalam (`table` -> `table_body` -> span
    ber-`html`), bukan di `lines` blok induknya. Versi sebelumnya hanya membaca tingkat
    teratas, dan akibatnya seluruh isi tabel hilang: tujuh tabel di dokumen SPPA keluar
    kosong, lalu dirender sebagai gambar. Layanan OCR tidak kena masalah ini karena ia
    memakai markdown buatan MinerU sendiri.
    """
    bagian = []
    for baris in blok.get("lines", []):
        for span in baris.get("spans", []):
            teks = span.get("content") or span.get("html") or ""
            if teks:
                bagian.append(teks)
    for anak in blok.get("blocks", []):
        teks = isi_blok(anak)
        if teks:
            bagian.append(teks)
    return " ".join(bagian).strip()


def main(berkas):
    t_awal = time.perf_counter()
    from mineru.cli.common import do_parse, read_fn

    pdf_bytes = read_fn(Path(berkas))

    import pymupdf

    jumlah = pymupdf.open(stream=pdf_bytes, filetype="pdf").page_count
    kirim(
        t="mulai",
        mesin="mineru",
        detik_muat=round(time.perf_counter() - t_awal, 2),
        halaman=jumlah,
    )

    t_ocr = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(jumlah):
            kirim(t="halaman_mulai", n=i + 1)
            t = time.perf_counter()
            do_parse(
                output_dir=tmp,
                pdf_file_names=[f"hal{i}"],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=["ch"],  # tidak ada kode "id"/"en"; model ch yang menangani Latin
                backend=BACKEND,
                start_page_id=i,
                end_page_id=i,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
            )

            tengah = next(Path(tmp).glob(f"hal{i}/**/*middle.json"))
            info = json.loads(tengah.read_text())["pdf_info"][0]
            lebar, tinggi = info["page_size"]

            for blok in info.get("para_blocks") or info.get("preproc_blocks") or []:
                x1, y1, x2, y2 = blok["bbox"]
                kirim(
                    t="blok",
                    n=i + 1,
                    kategori=blok.get("type", "text"),
                    bbox=[
                        round(x1 / lebar * SKALA),
                        round(y1 / tinggi * SKALA),
                        round(x2 / lebar * SKALA),
                        round(y2 / tinggi * SKALA),
                    ],
                    teks=isi_blok(blok),
                )
            kirim(t="halaman_selesai", n=i + 1, detik=round(time.perf_counter() - t, 2))

    kirim(t="selesai", detik=round(time.perf_counter() - t_ocr, 2))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:  # subprocess: galat harus sampai ke UI, bukan hilang di stderr
        kirim(t="galat", pesan=f"{type(e).__name__}: {e}")
        sys.exit(1)
