#!/usr/bin/env python
"""Runner PaddleOCR-VL: gambar halaman -> baris JSON ke stdout.

Dijalankan sebagai subprocess dengan ../.venv-paddle/bin/python. Kontrak keluarannya
sama untuk semua engine:

    {"t": "mulai", "mesin": "paddleocr-vl", "detik_muat": 3.7}
    {"t": "blok", "n": 1, "kategori": "text", "bbox": [x1,y1,x2,y2], "teks": "..."}
    {"t": "halaman_selesai", "n": 1, "detik": 15.4}
    {"t": "selesai", "detik": 30.8}

bbox memakai skala 0-999 seperti keluaran Unlimited-OCR, supaya panel highlight di
app.py tidak perlu tahu engine mana yang sedang jalan.

Dipakai varian VL, bukan pipeline klasik PP-OCRv5/v6, karena PaddlePaddle di Mac
tidak punya perangkat GPU sama sekali (`paddle.device.get_available_device()` kosong;
'mps' dan 'metal' bukan device yang sah). Pada jalur VL, pengenalan teks dilempar ke
server mlx-vlm sehingga kena GPU Apple — meski deteksi layout tetap di CPU Paddle.

Butuh server yang hidup lebih dulu:
    .venv/bin/python -m mlx_vlm.server --port 8111 \
        --model PaddlePaddle/PaddleOCR-VL-1.6 --trust-remote-code
"""

import json
import os
import sys
import time

SKALA = 1000.0
SERVER = os.environ.get("PADDLE_VL_SERVER", "http://localhost:8111/")
NAMA_MODEL = os.environ.get("PADDLE_VL_MODEL", "PaddlePaddle/PaddleOCR-VL-1.6")


def kirim(**ev):
    print(json.dumps(ev, ensure_ascii=False), flush=True)


def main(berkas):
    t_awal = time.perf_counter()
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(
        vl_rec_backend="mlx-vlm-server",
        vl_rec_server_url=SERVER,
        vl_rec_api_model_name=NAMA_MODEL,
    )
    kirim(t="mulai", mesin="paddleocr-vl", detik_muat=round(time.perf_counter() - t_awal, 2))

    t_ocr = time.perf_counter()
    for nomor, path in enumerate(berkas, 1):
        t = time.perf_counter()
        hasil = pipeline.predict(path)[0]
        lebar, tinggi = hasil["width"], hasil["height"]

        for blok in hasil["parsing_res_list"]:
            x1, y1, x2, y2 = blok.bbox
            kirim(
                t="blok",
                n=nomor,
                kategori=blok.label or "text",
                bbox=[
                    round(x1 / lebar * SKALA),
                    round(y1 / tinggi * SKALA),
                    round(x2 / lebar * SKALA),
                    round(y2 / tinggi * SKALA),
                ],
                teks=str(blok.content or "").strip(),
            )
        kirim(t="halaman_selesai", n=nomor, detik=round(time.perf_counter() - t, 2))

    kirim(t="selesai", detik=round(time.perf_counter() - t_ocr, 2))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:  # subprocess: galat harus sampai ke UI, bukan hilang di stderr
        kirim(t="galat", pesan=f"{type(e).__name__}: {e}")
        sys.exit(1)
