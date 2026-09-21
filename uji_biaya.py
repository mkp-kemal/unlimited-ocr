#!/usr/bin/env python
"""Uji biaya massal: OCR -> ekstraksi field -> analisis risiko, untuk banyak dokumen.

Memakai fungsi yang sama dengan demo web (app.py), jadi angkanya sama dengan yang
terlihat di UI, dan setiap panggilan LLM tercatat ke hasil/biaya-token-llm.xlsx.

    .venv/bin/python uji_biaya.py ../uji-bulk/*.pdf
    .venv/bin/python uji_biaya.py dok.pdf --semua-model
    .venv/bin/python uji_biaya.py dok.pdf --model openai/gpt-5.6-luna google/gemini-3.8-flash
    .venv/bin/python uji_biaya.py dok.pdf --tanpa-ekstraksi
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

import app
from field import FIELD_BAWAAN
from llm import FILE_PROMPT_ANALISIS, MODEL_BAWAAN, MODEL_LLM

AKAR = Path(__file__).resolve().parent
CACHE_OCR = AKAR / "hasil" / "cache-ocr"
DIR_ANALISIS = AKAR / "hasil" / "analisis"


def habiskan(generator):
    """Fungsi UI mengalirkan pembaruan; di sini yang dibutuhkan hanya keadaan akhirnya."""
    akhir = None
    for akhir in generator:
        pass
    return akhir


def jumlah_halaman(dok):
    if dok.suffix.lower() != ".pdf":
        return 1
    import fitz

    with fitz.open(dok) as d:
        return d.page_count


def ocr(dok, mesin, dpi, max_tokens):
    """OCR dengan cache per (isi file, engine, DPI).

    Uji LLM sering diulang untuk dokumen yang sama — mis. membandingkan model — dan OCR
    memakan menit sementara hasilnya tidak berubah. Kunci memakai isi file, bukan nama,
    supaya file yang diganti dengan nama sama tidak memakai hasil lama.
    """
    kunci = hashlib.sha1(dok.read_bytes()).hexdigest()[:12]
    engine = mesin.split()[0].lower()
    path = CACHE_OCR / f"{dok.stem}-{kunci}-{engine}-{dpi}.md"
    if path.exists():
        return path.read_text(encoding="utf-8"), True

    akhir = habiskan(app.jalankan(str(dok), mesin, dpi, max_tokens, app.PROMPT))
    md = akhir[2]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    return md, False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dokumen", nargs="+", type=Path)
    p.add_argument("--mesin", default="Unlimited-OCR — MLX", choices=list(app.MESIN))
    # 300, bukan 200: pada SPPA, Unlimited-OCR looping di halaman 1 pada DPI 200 dan
    # lancar di 100 maupun 300 (diukur 2026-09-17).
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--model", nargs="+", default=MODEL_BAWAAN)
    p.add_argument("--semua-model", action="store_true")
    p.add_argument("--json", type=Path, default=AKAR / "contoh" / "hazard-api-contoh.json")
    p.add_argument("--prompt", type=Path, default=FILE_PROMPT_ANALISIS)
    p.add_argument("--field", nargs="+", default=FIELD_BAWAAN)
    p.add_argument("--tanpa-ekstraksi", action="store_true",
                   help="lewati ekstraksi field; hanya analisis (untuk mengukur penghematannya)")
    p.add_argument("--tanpa-scan", action="store_true",
                   help="analisis tanpa teks dokumen: hanya JSON hazard + system prompt")
    args = p.parse_args()

    model = [m for m, *_ in MODEL_LLM] if args.semua_model else args.model
    json_hazard = args.json.read_text(encoding="utf-8")
    prompt = args.prompt.read_text(encoding="utf-8")
    ringkasan = []

    for i, dok in enumerate(args.dokumen, 1):
        print(f"\n[{i}/{len(args.dokumen)}] {dok.name}", flush=True)
        halaman = [None] * jumlah_halaman(dok)
        # Tanpa scan dan tanpa ekstraksi, teks dokumen tidak dipakai sama sekali.
        if args.tanpa_scan and args.tanpa_ekstraksi:
            md = ""
        else:
            md, dari_cache = ocr(dok, args.mesin, args.dpi, args.max_tokens)
            print(f"  OCR {args.mesin} @ {args.dpi} dpi · {len(halaman)} halaman"
                  f"{' · dari cache' if dari_cache else ''}", flush=True)

        serah = {}
        if not args.tanpa_ekstraksi:
            _, _, status, _, serah = habiskan(
                app.jalankan_ekstraksi(md, args.field, model, str(dok), args.mesin, halaman))
            print(f"  ekstraksi: {status}", flush=True)

        biaya, teks, status, _ = habiskan(app.jalankan_analisis(
            json_hazard, "" if args.tanpa_scan else app.isi_scan_md(md), prompt, model, serah,
            str(dok), args.mesin, halaman))
        print(f"  analisis : {status}", flush=True)

        DIR_ANALISIS.mkdir(parents=True, exist_ok=True)
        (DIR_ANALISIS / f"{dok.stem}.md").write_text(f"# {dok.name}\n\n{teks}\n", encoding="utf-8")

        biaya.insert(0, "Dokumen", dok.name)
        ringkasan.append(biaya)

    tabel = pd.concat(ringkasan, ignore_index=True)
    kolom_angka = ["Token masuk", "Token keluar", "Biaya analisis (USD)",
                   "Biaya ekstraksi (USD)", "Total per dokumen (USD)"]
    for k in kolom_angka:
        tabel[k] = pd.to_numeric(tabel[k], errors="coerce")
    rata = (tabel.groupby("Model")[kolom_angka].mean()
            .assign(**{"Per 1.000 dokumen (USD)": lambda t: t["Total per dokumen (USD)"] * 1000})
            .sort_values("Total per dokumen (USD)"))

    pd.set_option("display.width", 220)
    print(f"\n=== Rata-rata per model · {len(args.dokumen)} dokumen ===")
    print(rata.round(6).to_string())
    print(f"\nHasil analisis per dokumen: {DIR_ANALISIS.relative_to(AKAR)}/")
    print("Tercatat di: hasil/biaya-token-llm.xlsx")


if __name__ == "__main__":
    main()
