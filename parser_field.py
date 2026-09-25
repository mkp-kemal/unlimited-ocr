"""Parser field tanpa LLM — penerus ke parser layanan.

Kodenya TIDAK ada di sini. Sumbernya `alat/kode_parser.py`, dibangkitkan ke
`layanan/inti.py`, lalu diimpor dari sana. Demo dan tool datamapan karena itu
memakai parser yang sama persis — dulu keduanya sempat berbeda hasilnya karena
kodenya disalin, bukan dibagi.

Ubah parser di `alat/kode_parser.py`, lalu jalankan `python3 alat/bangun_config.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layanan"))

from inti import (  # noqa: E402  — setelah sys.path disiapkan
    CENTANG,
    KAMUS,
    KOSONG,
    LABEL_UMUM,
    betulkan_centang,
    betulkan_y,
    blok_dari_mentah,
    cari,
    kolom_isian,
    tata_letak,
)

__all__ = ["CENTANG", "KAMUS", "KOSONG", "LABEL_UMUM", "betulkan_centang", "betulkan_y",
           "blok_dari_mentah", "cari", "kolom_isian", "tata_letak"]
