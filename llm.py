"""Panggilan LLM lewat OpenRouter: ambil field dari teks OCR, lalu analisis field itu.

OCR tetap berjalan lokal. LLM dipakai dua kali per dokumen, dan token serta biaya
keduanya diukur dengan cara yang sama (`_panggil`):

1. ambil_field — membaca teks OCR dan mengisi field yang diminta.
2. analisis    — menganalisis risiko dari JSON hazard + scan OCR, dipandu system prompt.
"""

import json
import re
import time
from pathlib import Path

import requests

from field import periksa

AKAR = Path(__file__).resolve().parent
URL = "https://openrouter.ai/api/v1/chat/completions"

# Daftar pendek model, disaring dari puluhan model tiga penyedia di OpenRouter.
#
# Skor: BenchLM (benchlm.ai), indeks gabungan 8 kategori, dibaca 2026-09-17.
# Sumber lain (llm-stats, ringkasan Artificial Analysis) saling bertentangan pada model
# yang sama, jadi dipakai satu sumber yang tabelnya utuh. Harga: API model OpenRouter,
# hari yang sama, USD per sejuta token masuk/keluar.
#
# Per penyedia diambil yang skornya tertinggi ("terbaik") dan yang skor-per-harganya
# paling tinggi ("hemat"). Google tidak punya "seimbang": model terbaiknya sudah murah.
MODEL_LLM = [
    # id OpenRouter                   label                     tingkat     skor  masuk keluar
    ("anthropic/claude-fable-5.1",   "Claude Fable 5.1",        "terbaik",  84.8, 10.00, 50.00),
    ("anthropic/claude-opus-5",      "Claude Opus 5",           "seimbang", 81.8,  5.00, 25.00),
    ("anthropic/claude-sonnet-5",    "Claude Sonnet 5",         "hemat",    69.9,  2.00, 10.00),
    ("openai/gpt-6-astra",           "GPT-6 Astra",             "terbaik",  82.9, 10.00, 50.00),
    ("openai/gpt-5.6-sol",           "GPT-5.6 Sol",             "seimbang", 80.7,  2.00, 10.00),
    ("openai/gpt-5.6-luna",          "GPT-5.6 Luna",            "hemat",    64.7,  0.20,  1.20),
    ("google/gemini-3.8-flash",      "Gemini 3.8 Flash",        "terbaik",  75.5,  0.75,  3.75),
    ("google/gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite",   "hemat",    58.8,  0.30,  2.50),
]
# Bawaan: satu model hemat per penyedia, supaya perbandingan biaya pertama murah.
MODEL_BAWAAN = ["openai/gpt-5.6-luna", "google/gemini-3.5-flash-lite", "anthropic/claude-sonnet-5"]

PROMPT_SISTEM = """Kamu mengambil nilai field dari teks hasil OCR dokumen asuransi.

Teks berasal dari OCR: bisa memuat salah baca, tabel HTML, dan penanda halaman
`<!-- halaman N -->`.

Untuk SETIAP field yang diminta, isi:
- nilai: isi field menurut teks. Nama field bisa berbeda dari label di dokumen —
  cocokkan artinya (mis. "nama lengkap" dengan "Nama Tertanggung"). Kalau field itu
  tidak ada atau kosong di dokumen, isi null. Jangan menebak nilai yang tidak tertulis.
- kutipan: salin PERSIS potongan teks OCR tempat nilai ditemukan, lengkap dengan salah
  ketiknya. Jangan diperbaiki. null bila nilai null.
- halaman: angka N dari penanda halaman tempat kutipan berada. null bila nilai null.
- catatan: isi hanya bila nilai kamu rapikan atau tafsirkan dari salah baca OCR,
  jelaskan singkat. Selain itu null.

Untuk kotak centang (☐ ☑ ✓ ☒), nilainya adalah pilihan yang tercentang. Kalau tidak ada
yang jelas tercentang, isi null.

Jawab HANYA dengan satu objek JSON, tanpa teks lain. Kuncinya nama field persis seperti
diminta:
{"<nama field>": {"nilai": ..., "kutipan": ..., "halaman": ..., "catatan": ...}}"""

# System prompt analisis disimpan sebagai .md, bukan string di kode, supaya bisa disunting
# tanpa menyentuh Python. Isinya jadi isi awal form system prompt di UI.
FILE_PROMPT_ANALISIS = AKAR / "prompt" / "analisis-risiko.md"
PROMPT_ANALISIS = FILE_PROMPT_ANALISIS.read_text(encoding="utf-8")


def baca_env():
    """Pembaca .env kecil — cukup untuk KUNCI=nilai dan komentar sebaris ` #...`."""
    nilai = {}
    path = AKAR / ".env"
    if not path.exists():
        return nilai
    for baris in path.read_text(encoding="utf-8").splitlines():
        baris = baris.strip()
        if not baris or baris.startswith("#") or "=" not in baris:
            continue
        kunci, _, isi = baris.partition("=")
        nilai[kunci.strip()] = re.split(r"\s+#", isi, maxsplit=1)[0].strip().strip("\"'")
    return nilai


def _json_dari_jawaban(teks):
    """Ambil objek JSON dari jawaban model.

    Tidak memakai response_format: dukungannya berbeda antar penyedia, dan di OpenRouter
    parameter yang tidak didukung bisa membuat permintaan ditolak. Sebagai gantinya
    model diminta menjawab JSON, dan pembungkus ```json dibuang di sini.
    """
    teks = re.sub(r"^```(?:json)?\s*|\s*```$", "", (teks or "").strip())
    try:
        return json.loads(teks)
    except json.JSONDecodeError:
        cocok = re.search(r"\{.*\}", teks, re.S)
        if cocok:
            return json.loads(cocok.group(0))
        raise


def _panggil(model, sistem, isi_user):
    """Satu permintaan ke OpenRouter. Mengembalikan (teks_jawaban, pemakaian).

    Angka token dan biaya diambil dari respons OpenRouter, bukan dihitung sendiri:
    tiap penyedia memecah teks jadi token dengan cara berbeda.
    """
    env = baca_env()
    kunci = env.get("OPEN_ROUTER_API_KEY")
    # Nama variabelnya memang "ENALBLE" di .env; dibaca apa adanya.
    if env.get("OPEN_ROUTER_ENALBLE", "true").lower() != "true":
        raise RuntimeError("LLM dimatikan: OPEN_ROUTER_ENALBLE bukan 'true' di .env.")
    if not kunci:
        raise RuntimeError("OPEN_ROUTER_API_KEY belum diisi di unlimited-ocr/.env.")

    t0 = time.perf_counter()
    respons = requests.post(
        URL,
        headers={"Authorization": f"Bearer {kunci}", "X-Title": "OCR Polis Demo"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": sistem},
                {"role": "user", "content": isi_user},
            ],
            "temperature": 0,
        },
        timeout=300,
    )
    detik = time.perf_counter() - t0

    data = respons.json()
    if respons.status_code != 200 or "error" in data:
        pesan = data.get("error", {}).get("message", respons.text[:300])
        raise RuntimeError(f"OpenRouter menolak ({respons.status_code}): {pesan}")

    u = data.get("usage", {})
    pemakaian = {
        "model_terpakai": data.get("model", model),
        "token_masuk": u.get("prompt_tokens", 0),
        "token_cache": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "token_keluar": u.get("completion_tokens", 0),
        # Token "berpikir" dibayar sebagai token keluaran walau tidak tampil di jawaban.
        # Sudah termasuk di token_keluar; dipisah supaya kelihatan porsinya.
        "token_berpikir": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        "total_token": u.get("total_tokens", 0),
        # Satuan OpenRouter adalah kredit, setara USD.
        "biaya_usd": u.get("cost", 0.0),
        "detik": round(detik, 2),
    }
    return data["choices"][0]["message"].get("content") or "", pemakaian


def ambil_field(teks_ocr, field, model):
    """Minta LLM mengisi field dari teks OCR. Mengembalikan (hasil, pemakaian, teks_mentah).

    `hasil` selalu memuat SEMUA field yang diminta — yang tidak dijawab model diisi null —
    dan tiap field sudah diberi status oleh `field.periksa`. Bila jawaban model bukan JSON
    yang terbaca, `hasil` bernilai None tapi `pemakaian` tetap dikembalikan: tokennya
    sudah terbayar dan harus tetap tercatat.
    """
    isi_user = (
        "Field yang diminta:\n" + "\n".join(f"- {f}" for f in field)
        + f"\n\nTeks OCR:\n<<<\n{teks_ocr}\n>>>"
    )
    teks, pemakaian = _panggil(model, PROMPT_SISTEM, isi_user)
    try:
        jawaban = _json_dari_jawaban(teks)
    except (json.JSONDecodeError, ValueError):
        return None, pemakaian, teks

    # Model kadang mengubah huruf besar-kecil nama field; dicocokkan tanpa memedulikannya.
    per_kunci = {str(k).casefold(): v for k, v in jawaban.items() if isinstance(v, dict)}
    hasil = {}
    for f in field:
        isi = per_kunci.get(f.casefold()) or {}
        hasil[f] = {k: isi.get(k) for k in ("nilai", "kutipan", "halaman", "catatan")}
    return periksa(hasil, teks_ocr), pemakaian, teks


def analisis(json_hazard, scan_md, system_prompt, model):
    """Analisis risiko dari tiga masukan. Mengembalikan (teks_jawaban, pemakaian).

    System prompt dikirim sebagai pesan sistem, dan JSON hazard serta scan OCR sebagai
    pesan pengguna. Dengan begitu bagian depan permintaan — system prompt — sama persis
    di setiap dokumen, dan bagian itulah yang bisa di-cache penyedia model (kolom
    token_cache). Jawabannya teks biasa untuk dibaca orang, jadi tidak diurai sebagai JSON.
    """
    bagian = []
    if (json_hazard or "").strip():
        bagian.append(f"## Data hazard (JSON)\n\n```json\n{json_hazard.strip()}\n```")
    if (scan_md or "").strip():
        bagian.append(f"## Scan dokumen (hasil OCR)\n\n<<<\n{scan_md.strip()}\n>>>")
    return _panggil(model, system_prompt.strip(), "\n\n".join(bagian))
