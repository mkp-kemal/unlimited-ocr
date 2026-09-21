"""Parser field tanpa LLM: cari nilai field dari blok OCR beserta posisinya.

Bekerja dari output MENTAH engine, bukan markdown, karena hanya di sana koordinat tiap
blok masih ada: tag `<|det|>kategori [bbox]<|/det|>teks` dari Unlimited-OCR, atau baris
JSON `{"kategori", "bbox", "teks"}` dari MinerU dan PaddleOCR-VL. Ketiganya memakai bbox
skala 0-999, jadi satu parser berlaku untuk semua engine.

Untuk setiap field, label dicari lewat kamus sebutannya, lalu nilainya diambil dengan
cara pertama yang berhasil:

1. baris  : "Label : nilai" dalam satu baris teks atau satu kotak tabel.
2. tabel  : label di satu kotak, nilai di kotak berikutnya pada baris tabel yang sama.
3. posisi : label berdiri sendiri; nilai = blok terdekat di kanannya pada baris yang
            sama, atau tepat di bawahnya. Untuk tata letak yang label dan isinya keluar
            sebagai blok terpisah — sering terjadi pada Unlimited-OCR dan MinerU.
4. pola   : cadangan untuk NIK dan nomor HP bila labelnya tidak ketemu, hanya kalau
            polanya muncul tepat sekali di dokumen.
"""

import difflib
import html
import json
import re
import unicodedata
from dataclasses import dataclass

from ocr import blok_det

# Sebutan lain tiap field dan jenis nilainya. Field yang tidak ada di sini tetap dicari,
# dengan namanya sendiri sebagai satu-satunya sebutan.
KAMUS = {
    "nama lengkap": (["nama lengkap", "nama tertanggung", "nama pemohon", "nama nasabah"], "teks"),
    "nomor identitas": (["nomor identitas", "nik", "nomor ktp", "nomor induk kependudukan"], "nik"),
    "tempat tanggal lahir": (["tempat tanggal lahir", "ttl"], "teks"),
    "alamat lengkap": (["alamat lengkap", "alamat lengkap ktp", "alamat"], "teks"),
    "nomor hp": (["nomor telp hp", "nomor hp", "nomor telepon", "telepon", "handphone"], "telepon"),
    "jumlah pertanggungan": (["jumlah pertanggungan", "nilai pertanggungan",
                              "total pertanggungan", "harga pertanggungan"], "rupiah"),
    "jangka waktu pertanggungan": (["jangka waktu pertanggungan", "periode pertanggungan",
                                    "masa pertanggungan"], "teks"),
    "nomor polis": (["nomor polis"], "teks"),
    "total premi dibayar": (["total premi dibayar", "total premi"], "rupiah"),
}

# Label yang lazim di formulir tapi belum tentu dicari. Dikenali supaya tidak pernah
# terambil sebagai NILAI field lain — pada uji SPPA asli, "Kota" sempat berisi "Provinsi".
LABEL_UMUM = [
    "provinsi", "kota", "kabupaten", "kecamatan", "kelurahan", "kode pos", "rt rw",
    "jenis kelamin", "agama", "kewarganegaraan", "status perkawinan", "pekerjaan", "email",
    "npwp", "nomor npwp", "sumber dana", "status tertanggung", "tujuan transaksi",
    "beneficial owner", "nama perusahaan", "alamat domisili", "nomor kartu keluarga",
    "hubungan dalam keluarga", "rata rata penghasilan", "nomor telp pic", "email pic",
    "nama pic", "nomor izin usaha", "alamat perusahaan",
]

# Field pilihan: kalau tanda centang tidak ikut terbaca OCR, semua opsinya muncul
# bersamaan ("Laki-laki Perempuan"), dan parser harus menolak menebak.
OPSI = {"jenis kelamin": ["laki laki", "perempuan"]}

# Sebutan yang lebih panjang dari ini hampir pasti kalimat, bukan label. Tanpa batas ini,
# "alamat" akan cocok dengan kalimat pasal yang kebetulan memuat kata alamat.
MAKS_KATA_LABEL = 7

CENTANG = "☑✓✔☒■"
KOSONG = "☐□"
RE_KOTAK = re.compile(rf"([{CENTANG}{KOSONG}])\s*([^{CENTANG}{KOSONG}]+)")
RE_HALAMAN = re.compile(r"<!-- halaman (\d+) -->")
RE_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
RE_TD = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
# "Label : nilai" — sebelum titik dua harus ada huruf, dan tidak terlalu panjang.
RE_BERLABEL = re.compile(r"^\s*([^:]{2,60}?)\s*:\s*(.*)$")
POLA = {
    "nik": re.compile(r"(?<!\d)\d{16}(?!\d)"),
    "telepon": re.compile(r"(?<!\d)(?:\+62|62|0)8\d{7,11}(?!\d)"),
}


@dataclass
class Satuan:
    """Satu baris teks, atau satu baris tabel, beserta perkiraan kotaknya di halaman."""

    halaman: int
    bbox: tuple
    teks: str
    sel: list = None  # kotak-kotak tabel, bila satuan ini baris tabel
    blok: int = 0  # nomor blok asalnya — baris-baris satu blok saling bersaudara


def _norm(s):
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    kata = ["nomor" if k in ("no", "nmr", "nom") else k for k in s.split()]
    return " ".join(kata)


def _bersih(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def _hanya_tanda(s):
    return not s or re.fullmatch(r"[\s:\-–—|.,;]+", s) is not None


def blok_dari_mentah(mentah):
    """Output mentah panel -> daftar (halaman, kategori, bbox, teks)."""
    potong = RE_HALAMAN.split(mentah or "")
    hasil = []
    for n, isi in zip(potong[1::2], potong[2::2]):
        if "<|det|>" in isi:
            hasil += [(int(n), k, tuple(b), t) for k, b, t in blok_det(isi)[1]]
            continue
        for baris in isi.splitlines():
            if not baris.strip().startswith("{"):
                continue
            try:
                d = json.loads(baris)
            except json.JSONDecodeError:
                continue
            hasil.append((int(n), d.get("kategori", ""), tuple(d.get("bbox") or (0, 0, 0, 0)),
                          d.get("teks", "")))
    return hasil


def _satuan(blok):
    """Pecah blok jadi baris. Koordinat per baris hanya PERKIRAAN: engine memberi kotak per
    blok, jadi tinggi blok dibagi rata ke jumlah barisnya."""
    for id_blok, (n, _, (x1, y1, x2, y2), teks) in enumerate(blok):
        if "<table" in (teks or "").lower():
            baris = [[_bersih(c) for c in RE_TD.findall(tr)] for tr in RE_TR.findall(teks)]
            baris = [b for b in baris if any(b)]
            tinggi = (y2 - y1) / max(len(baris), 1)
            for i, sel in enumerate(baris):
                yield Satuan(n, (x1, y1 + i * tinggi, x2, y1 + (i + 1) * tinggi),
                             " ".join(s for s in sel if s), sel, id_blok)
        else:
            baris = [_bersih(b).strip("#*> ") for b in (teks or "").splitlines()]
            baris = [b for b in baris if b]
            tinggi = (y2 - y1) / max(len(baris), 1)
            for i, b in enumerate(baris):
                yield Satuan(n, (x1, y1 + i * tinggi, x2, y1 + (i + 1) * tinggi), b,
                             blok=id_blok)


def _calon_label(s):
    """Bagian satuan yang mungkin label: (teks label, nilai sebaris, indeks kotak)."""
    potongan = enumerate(s.sel) if s.sel is not None else [(None, s.teks)]
    for i, teks in potongan:
        m = RE_BERLABEL.match(teks)
        if m and re.search(r"[A-Za-z]", m.group(1)):
            yield m.group(1), m.group(2).strip(), i
        else:
            yield teks, "", i


def _cocok(label, nama_norm, sebutan):
    """(peringkat, skor, cara) bila label cocok; peringkat kecil = lebih meyakinkan."""
    lab = _norm(label)
    if not lab or len(lab.split()) > MAKS_KATA_LABEL:
        return None
    if lab == nama_norm:
        return 0, 1.0, "persis"
    if lab in sebutan:
        return 1, 1.0, "sebutan"
    kata_lab = set(lab.split())
    terbaik = None
    for s in sebutan:
        kata = set(s.split())
        # Dicocokkan per kata utuh, bukan potongan huruf: "tangga" tidak boleh cocok
        # dengan "tanggal".
        if kata <= kata_lab:
            skor = len(kata) / len(kata_lab)
            if not terbaik or skor > terbaik[1]:
                terbaik = (2, skor, "sebagian")
    if terbaik:
        return terbaik
    # Mirip dinilai PER KATA dan jumlah katanya harus sama: salah ketik OCR
    # ("Tentanggung") lolos, tapi "nomor telp pic" tidak cocok dengan "nomor telp hp".
    terbaik = 0.0
    for s in sebutan:
        a, b = lab.split(), s.split()
        if len(a) == len(b):
            rasio = min(difflib.SequenceMatcher(None, x, y).ratio() for x, y in zip(a, b))
            terbaik = max(terbaik, rasio)
    return (3, terbaik, "mirip") if terbaik >= 0.8 else None


def _pilih_centang(teks):
    """'☐ Baru ☑ Perpanjangan' -> 'Perpanjangan'. None bila tak satu pun tercentang."""
    kotak = RE_KOTAK.findall(teks)
    if not kotak:
        return teks
    pilih = [isi.strip() for tanda, isi in kotak if tanda in CENTANG]
    return ", ".join(pilih) or None


def _nilai_dari_teks(teks):
    teks = teks.strip().lstrip(":|").strip()
    if _hanya_tanda(teks):
        return None
    return _pilih_centang(teks)


def _nilai_baris_tabel(sel, i):
    """Nilai = kotak bermakna pertama setelah kotak label. Berhenti di kotak yang ternyata
    label lain (mis. 'No. Identitas: …', 'Email:') — itu milik field lain."""
    centang = []
    for s in sel[i + 1:]:
        if _hanya_tanda(s):
            continue
        if RE_BERLABEL.match(s) and re.search(r"[A-Za-z]", s.split(":")[0]):
            break
        if any(c in s for c in CENTANG + KOSONG):
            centang.append(s)
            continue
        if centang:
            break
        return _nilai_dari_teks(s)
    return _pilih_centang(" ".join(centang)) if centang else None


def _tumpang(a1, a2, b1, b2):
    return max(0.0, min(a2, b2) - max(a1, b1))


def _nilai_posisi(label, semua, adalah_label):
    """Nilai = satuan terdekat di kanan label pada baris yang sama; bila tak ada, tepat di
    bawahnya. Satuan yang sendirinya label field lain dilewati."""
    x1, y1, x2, y2 = label.bbox
    tinggi = max(y2 - y1, 1)
    sehalaman = [s for s in semua if s is not label and s.halaman == label.halaman
                 and not adalah_label(s.teks)]
    kanan = [s for s in sehalaman
             if s.bbox[0] >= x2 - 5
             and _tumpang(y1, y2, s.bbox[1], s.bbox[3]) >= 0.5 * min(tinggi, s.bbox[3] - s.bbox[1])]
    # "Di bawah" tidak boleh dari blok yang sama: pada daftar label vertikal, baris di
    # bawah sebuah label adalah label berikutnya, bukan nilainya.
    bawah = [s for s in sehalaman
             if s.blok != label.blok
             and 0 <= s.bbox[1] - y2 + 2 <= 1.5 * tinggi
             and _tumpang(x1, x2, s.bbox[0], s.bbox[2]) > 0]
    for kumpulan, jarak in ((kanan, lambda s: s.bbox[0] - x2), (bawah, lambda s: s.bbox[1] - y2)):
        for s in sorted(kumpulan, key=jarak):
            nilai = (_nilai_baris_tabel([""] + s.sel, 0) if s.sel is not None
                     else _nilai_dari_teks(s.teks))
            if nilai:
                return nilai
    return None


def _layak(nilai, nama, jenis, frasa_label):
    """Saring nilai calon. Mengembalikan nilai yang sudah dipangkas, atau None bila ditolak.

    - Ditolak bila nilainya diawali label field LAIN: yang terambil adalah baris lain.
    - Dipangkas di label yang menempel di tengah ("VANGKALAN KERINCI Kode Pos"). Hanya
      label dua kata atau lebih, atau yang diikuti titik dua: label satu kata terlalu
      mudah bentrok dengan isi sungguhan, mis. kelurahan "PANGKALAN KERINCI KOTA".
    - Untuk NIK dan nomor HP, bentuk yang salah ditolak — lebih baik kosong daripada
      alamat yang tercatat sebagai nomor HP.
    - Field pilihan yang semua opsinya muncul tanpa tanda centang ditolak.
    """
    kata = nilai.split()
    norm = [(i, _norm(k)) for i, k in enumerate(kata) if _norm(k)]
    urutan = [w for _, w in norm]
    for posisi, (asli, _) in enumerate(norm):
        for frasa in frasa_label:
            if tuple(urutan[posisi:posisi + len(frasa)]) != frasa or " ".join(frasa) == nama:
                continue
            if posisi == 0:
                return None
            titik_dua = kata[norm[posisi + len(frasa) - 1][0]].endswith(":")
            if len(frasa) >= 2 or titik_dua:
                nilai = " ".join(kata[:asli]).strip(" :|,;")
                return nilai or None
    if jenis in POLA and not POLA[jenis].search(re.sub(r"[\s.\-]", "", nilai)):
        return None
    opsi = OPSI.get(nama)
    if opsi and sum(o in _norm(nilai) for o in opsi) > 1:
        return None
    return nilai


def _periksa_bentuk(nilai, jenis):
    angka = re.sub(r"\D", "", nilai)
    if jenis == "nik" and len(angka) != 16:
        return f"NIK biasanya 16 digit, terbaca {len(angka)}"
    if jenis == "telepon" and not re.fullmatch(r"(62|0)8\d{7,11}", angka):
        return "bentuk nomor HP tidak lazim"
    if jenis == "rupiah" and not angka:
        return "tidak ada angka nominal"
    return ""


def cari(mentah, diminta):
    """Output mentah panel + daftar nama field -> {field: hasil}."""
    semua = sorted(_satuan(blok_dari_mentah(mentah)),
                   key=lambda s: (s.halaman, s.bbox[1], s.bbox[0]))
    semua_sebutan = {_norm(s) for sebutan, _ in KAMUS.values() for s in sebutan}
    semua_sebutan |= {_norm(f) for f in diminta} | {_norm(s) for s in LABEL_UMUM}
    frasa_label = sorted({tuple(s.split()) for s in semua_sebutan if s},
                         key=len, reverse=True)

    def adalah_label(teks):
        return _norm(teks.rstrip(": ")) in semua_sebutan

    hasil = {}
    for f in diminta:
        nama = _norm(f)
        sebutan, jenis = KAMUS.get(nama, ([nama], "teks"))
        sebutan = [_norm(s) for s in sebutan] + ([nama] if nama not in sebutan else [])

        calon = []
        for s in semua:
            for label, sebaris, i in _calon_label(s):
                m = _cocok(label, nama, sebutan)
                if m:
                    calon.append((m[0], -m[1], s.halaman, s.bbox[1], m[2], s, label, sebaris, i))
        calon.sort(key=lambda c: c[:4])

        def saring(n):
            return _layak(n, nama, jenis, frasa_label) if n else None

        pilihan = None
        for _, _, _, _, cara, s, label, sebaris, i in calon:
            nilai, sumber = saring(_nilai_dari_teks(sebaris) if sebaris else None), "baris"
            # Kotak berisi dua label ("Tempat/ Tanggal Lahir : Jenis Kelamin"): kotak
            # sebelahnya milik label yang kedua, jadi jangan diambil.
            sisa = _nilai_dari_teks(sebaris) if sebaris else None
            dua_label = bool(sisa) and _layak(sisa, "", "teks", frasa_label) is None
            if not nilai and not dua_label and s.sel is not None and i is not None:
                nilai, sumber = saring(_nilai_baris_tabel(s.sel, i)), "tabel"
            # Posisi hanya untuk label di luar tabel. Label di dalam tabel nilainya di
            # baris yang sama; kalau kotaknya kosong, baris di bawahnya milik field lain.
            if not nilai and s.sel is None:
                nilai, sumber = saring(_nilai_posisi(s, semua, adalah_label)), "posisi"
            if nilai:
                pilihan = {"nilai": nilai, "label_ditemukan": label, "cara": cara,
                           "sumber": sumber, "halaman": s.halaman}
                break

        if pilihan is None and jenis in POLA:
            temuan = {m.group(0) for s in semua for m in POLA[jenis].finditer(s.teks)}
            if len(temuan) == 1:
                pilihan = {"nilai": temuan.pop(), "label_ditemukan": "", "cara": "pola",
                           "sumber": "pola", "halaman": None}

        if pilihan is None:
            alasan = "label ditemukan, nilainya kosong" if calon else "label tidak ditemukan"
            hasil[f] = {"nilai": None, "alasan": alasan,
                        "label_ditemukan": calon[0][6] if calon else ""}
            continue
        pilihan["catatan"] = _periksa_bentuk(pilihan["nilai"], jenis)
        hasil[f] = pilihan
    return hasil
