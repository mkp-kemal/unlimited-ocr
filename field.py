"""Field yang dicari, teks OCR yang dikirim ke LLM, dan pemeriksaan jawaban LLM.

Nilai field diambil oleh LLM dari teks hasil OCR, bukan dengan aturan pencocokan label.
Karena LLM bisa "membetulkan" salah baca OCR — atau mengarang — setiap nilai wajib
disertai kutipan persis dari teks OCR, dan kutipan itu diperiksa di sini.
"""

import html
import re

# Nama field ditulis bahasa biasa. LLM mencocokkan arti, jadi tidak perlu sama persis
# dengan label di dokumen ("nama lengkap" menemukan "Nama Tertanggung").
FIELD_BAWAAN = [
    "nama lengkap",
    "nomor identitas",
    "tempat tanggal lahir",
    "alamat lengkap",
    "nomor hp",
    "jumlah pertanggungan",
    "jangka waktu pertanggungan",
]

# Potongan gambar disisipkan sebagai data URI untuk tampilan. Isinya base64 yang panjang:
# pada SPPA 3 halaman, ini yang membuat markdown membengkak jadi >100 ribu karakter.
# Dikirim ke LLM, itu hanya membayar token untuk sesuatu yang tidak bisa dibaca.
RE_GAMBAR_DATA = re.compile(r"!\[[^\]]*\]\(data:[^)]*\)")
RE_TAG = re.compile(r"<[^>]+>")


def teks_untuk_llm(markdown):
    """Markdown panel -> teks yang dikirim ke LLM: tanpa gambar, penanda halaman tetap."""
    teks = RE_GAMBAR_DATA.sub("", markdown or "")
    return re.sub(r"\n{3,}", "\n\n", teks).strip()


def _rata(s):
    """Samakan bentuk teks untuk dibandingkan: tanpa tag HTML, entitas, dan beda spasi."""
    s = html.unescape(RE_TAG.sub(" ", str(s or "")))
    return re.sub(r"\s+", " ", s).strip().casefold()


def periksa(hasil, teks_ocr):
    """Tandai tiap field dengan status, berdasarkan kutipan yang dikembalikan LLM.

    - tidak ditemukan : LLM menjawab nilai kosong.
    - tidak terbukti  : kutipannya tidak ada di teks OCR — nilainya tidak bisa dipercaya.
    - ditafsirkan     : kutipan ada, tapi nilainya tidak tertulis persis di kutipan itu;
                        biasanya LLM membetulkan salah baca OCR. Mungkin benar, tetap tebakan.
    - terbukti        : kutipan ada di teks OCR dan nilainya tertulis di dalamnya.
    """
    sumber = _rata(teks_ocr)
    for isi in hasil.values():
        nilai, kutipan = isi.get("nilai"), isi.get("kutipan")
        if nilai in (None, ""):
            isi["status"] = "tidak ditemukan"
        elif not kutipan or _rata(kutipan) not in sumber:
            isi["status"] = "tidak terbukti"
        elif _rata(nilai) not in _rata(kutipan):
            isi["status"] = "ditafsirkan"
        else:
            isi["status"] = "terbukti"
    return hasil
