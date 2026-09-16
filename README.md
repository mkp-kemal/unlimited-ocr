# Unlimited-OCR di Apple Silicon (MLX)

OCR satu file (gambar atau PDF) pakai model `baidu/Unlimited-OCR` (3B, VLM end-to-end),
dijalankan lewat MLX supaya memakai GPU Apple.

Ada dua cara pakai: CLI [`ocr.py`](ocr.py) yang menyimpan hasil ke `hasil/`, dan demo web
[`app.py`](app.py) yang menampilkan proses bacanya secara realtime. Demo web-nya juga
bisa membandingkan tiga engine — Unlimited-OCR, MinerU, dan PaddleOCR-VL — dalam
tampilan yang sama.

## Kenapa MLX, bukan kode resmi Baidu

Repo resmi hanya menyediakan jalur NVIDIA: kode `transformers`-nya hardcode `.cuda()`,
dan dua opsi serving-nya (vLLM, SGLang dengan backend FlashAttention-3) CUDA-only.
Docker juga tidak menolong di Mac, karena container Linux tidak bisa mengakses GPU Apple.

Jalur PyTorch MPS dengan patch juga tidak dipakai di sini: bobot bf16 ~6,2 GB, dan fork
komunitas memaksa fp32 di MPS karena bf16 menghasilkan output rusak — itu ~12,4 GB bobot
saja, terlalu sempit di mesin 16 GB.

Yang dipakai: `mlx-vlm`, yang sejak 0.6.4 punya dukungan arsitektur `unlimited_ocr`
native. Penting, port MLX ini mengimplementasikan `RingSlidingKVCache`
(`mlx_vlm/models/unlimited_ocr/language.py`), jadi Reference Sliding Window Attention —
kontribusi utama paper, yang membuat KV cache konstan — tetap ada, bukan hilang jadi
decoder DeepSeek-OCR biasa.

Model: `mlx-community/Unlimited-OCR-8bit` (3,66 GB, kuantisasi 8-bit affine group size 64).

## Setup

Sudah dilakukan sekali di folder ini; untuk mengulang dari nol:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "mlx-vlm>=0.6.8" pymupdf gradio
```

Unduhan model (3,66 GB) terjadi otomatis saat run pertama, disimpan di `~/.cache/huggingface`.

## Pakai lewat CLI

```bash
# gambar
.venv/bin/python ocr.py ../dataset/sumber/contoh-kwitansi.jpeg

# PDF, halaman 1-3 saja
.venv/bin/python ocr.py ../uji-bulk/polis-001.pdf --halaman 1-3
```

Opsi: `--out` (folder keluaran, default `hasil`), `--dpi` (render PDF, default 200),
`--halaman 1-5`, `--max-tokens` (default 8192), `--prompt`, `--model`.

Keluaran per file input:

- `hasil/<nama>.md` — keluaran mentah model, lengkap dengan tag layout dan bounding box
- `hasil/<nama>.bersih.md` — versi Markdown yang enak dibaca, tag dibuang
- `hasil/<nama>-gambar/halN-01.png` — potongan blok gambar dari dokumen (logo, stempel,
  tanda tangan), dirujuk otomatis dari `.bersih.md` sehingga ikut tampil di preview
- `hasil/<nama>.json` — waktu muat model, total detik, dan per halaman: detik, jumlah
  token, token/detik, jumlah karakter

Progres dan ringkasan waktu dicetak ke stderr, teks hasil ke stdout.

## Demo web (Gradio)

Gradio jalan sepenuhnya lokal — tidak ada data yang keluar dari mesin ini.

Dua proses harus hidup, dan **keduanya mati setiap laptop direstart**: app Gradio di
:7860 dan server mlx-vlm di :8111 (dipakai PaddleOCR-VL). Skrip ini menyalakan yang
belum jalan:

```bash
./jalankan.sh          # nyalakan keduanya (idempoten, aman diulang)
./jalankan.sh stop     # matikan keduanya
```

Kalau ingin manual:

```bash
.venv/bin/python app.py                       # Ctrl-C untuk berhenti
nohup .venv/bin/python app.py > /tmp/gradio-ocr.log 2>&1 &   # latar belakang
```

Buka **http://127.0.0.1:7860**, unggah PDF/gambar, pilih engine, klik **Jalankan OCR**.

### Tiga engine dan device-nya

| Engine | Device | Waktu/halaman | Cara kerja |
| --- | --- | --- | --- |
| Unlimited-OCR | MLX (GPU) | ~8 s | VLM, streaming asli per token |
| MinerU | MPS (GPU) | ~12 s | pipeline deteksi+rekognisi, hasil per halaman |
| PaddleOCR-VL | MLX (GPU) + CPU | ~19 s | VLM lewat server mlx-vlm; layout tetap CPU |

**PaddleOCR-VL perlu server terpisah dinyalakan lebih dulu:**

```bash
.venv/bin/python -m mlx_vlm.server --port 8111 \
  --model PaddlePaddle/PaddleOCR-VL-1.6 --trust-remote-code
```

Matikan dengan `kill $(lsof -ti tcp:8111)`. Kalau server ini mati — dan itu yang terjadi
setiap laptop direstart — engine PaddleOCR-VL gagal sementara dua engine lain tetap jalan.
App memeriksanya lebih dulu dan menampilkan pesan berisi perintah untuk menyalakannya,
bukan galat mentah. Cara paling ringkas: `./jalankan.sh`.

Kenapa bukan PaddleOCR klasik (PP-OCRv5/v6): PaddlePaddle di Mac tidak punya perangkat
GPU sama sekali — `paddle.device.get_available_device()` mengembalikan `[]`, dan `mps`
maupun `metal` bukan nilai device yang sah. Satu-satunya jalur GPU untuk Paddle di sini
adalah varian VL, yang melempar pengenalan teks ke MLX.

Kenapa MinerU tidak pakai jalur MLX-nya sendiri: MinerU 3.4.4 punya client MLX, tapi
mengaktifkannya butuh `mlx-vlm` di `.venv-mineru`, sedangkan `mlx-vlm` menuntut
`transformers` 5.x dan MinerU mensyaratkan `<5.0.0`. MPS sudah terbukti bekerja
(10 s vs 14 s di CPU, hasil identik), jadi itu yang dipakai.

### Soal kata "stream"

Hanya Unlimited-OCR yang benar-benar streaming: token keluar satu per satu dan tag
`<|det|>` berisi bbox muncul sebelum teks bloknya, jadi highlight bisa mendahului.
MinerU dan PaddleOCR-VL baru mengirim hasil setelah satu halaman rampung; bloknya
diputar berjarak `JEDA_BLOK` (0,1 detik) supaya alurnya terbaca. Itu tampilan, bukan
kecepatan asli — angka detik di status tetap waktu sebenarnya.

Mematikan:

```bash
lsof -nP -iTCP:7860 -sTCP:LISTEN   # lihat dulu siapa yang pegang portnya
kill $(lsof -ti tcp:7860)          # matikan proses itu
```

Patokannya port, bukan nama proses: command line-nya cuma `.venv/bin/python app.py`,
jadi `pkill -f app.py` bisa ikut membunuh app.py milik proyek lain.

Ganti port lewat baris `demo.launch(...)` di akhir `app.py` kalau 7860 bentrok.

Tiga panel, semuanya bergerak bersamaan saat model membaca:

1. **Dokumen** — halaman dengan highlight blok. Biru = sudah dibaca, oranye tebal = sedang
   dibaca. Ini mungkin karena tag `<|det|>` berisi bounding box keluar *sebelum* teks
   bloknya, jadi kotak bisa digambar lebih dulu.
2. **Output mentah** — token mengalir apa adanya, lengkap dengan tag dan koordinat.
3. **Markdown** — hasil konversi, termasuk potongan gambar dari dokumen yang disisipkan
   sebagai data URI.

Catatan performa: panel teks disegarkan tiap 8 token dan gambar hanya digambar ulang saat
ada blok baru. Menyegarkan tiap token membuat UI tersendat karena menyusun ulang overlay
jauh lebih mahal daripada mengirim teks. Halaman pertama menunggu model dimuat (~4 detik);
setelah itu model tetap di memori selama server hidup.

## Kecepatan terukur (M4, 16 GB)

| Uji | Hasil |
| --- | --- |
| Muat model (sudah tercache) | 4,4 s |
| Render PDF 2 halaman @ 200 dpi | 0,3 s |
| PDF polis, 2 halaman | 13,5 s total — 6,7 s/halaman, ~600 token/halaman |
| Kwitansi JPEG kecil | 8,8 s |

Unduhan model pertama kali (3,66 GB) makan waktu sendiri, sekitar 30 menit di koneksi uji.

## Yang perlu diketahui

- **Keluarannya bukan Markdown.** Model mengeluarkan format layout sendiri:
  `<|det|>kategori [x1, y1, x2, y2]<|/det|>` diikuti isi bloknya. Di preview Markdown, tag
  itu muncul apa adanya sebagai teks karena `<|det|>` bukan tag HTML yang sah — karakter
  `|` membuatnya gagal diparse sebagai elemen. Yang ter-render hanya blok `<table>`, karena
  itu memang HTML valid. Karena itu skrip menulis dua file: `.md` mentah untuk yang butuh
  kategori blok dan koordinat (misalnya ekstraksi field berbasis posisi), dan `.bersih.md`
  hasil konversi — `title` jadi `##`, `page_number` jadi miring, `<table>` dibiarkan utuh
  (sudah diuji: tabel HTML dan gambar base64 memang ter-render di browser).
- **Koordinat bbox berskala 0-999, bukan piksel.** Semua pemetaan ke piksel halaman
  (highlight dan potongan gambar) memakai skala itu; lihat `SKALA` di `ocr.py`.
- **Gambar kecil otomatis diperbesar.** Gambar dengan sisi panjang di bawah 1800 px
  di-resize LANCZOS lebih dulu. Ini bukan kosmetik: pada kwitansi 600x227 px, versi asli
  melewatkan blok judul sepenuhnya. Angka 1800 disetel dari satu sampel — menaikkan sisi
  pendek ke 1024 (setara 4,5x) malah membuat judul itu hilang lagi, sedangkan 3x
  menangkapnya. Kalau hasil di dokumen lain terasa meleset, angka ini kandidat pertama
  untuk disetel ulang.
- **Per halaman, bukan satu forward pass.** Skrip ini meng-OCR halaman satu per satu.
  Kemampuan puluhan halaman sekali jalan (prompt `Multi page parsing.`) belum diuji di
  jalur MLX ini.
- **Tidak ada `no_repeat_ngram`.** Kode resmi memakai `no_repeat_ngram_size=35,
  ngram_window=128` untuk mencegah model terjebak mengulang teks; `mlx-vlm` tidak punya
  parameter itu. Yang tersedia hanya `repetition_penalty`, yang justru merusak OCR karena
  dokumen memang wajar mengandung kata berulang, jadi tidak dipakai. Loop semacam itu
  pernah terlihat sekali — lewat CLI mentah `python -m mlx_vlm.generate` pada gambar
  600x227 px, model mengulang satu frasa sampai batas token. Lewat `ocr.py` pada gambar
  yang sama hal itu tidak terulang. Kalau suatu saat muncul, batasi dengan `--max-tokens`.
- **Tulisan tangan tetap sulit.** Pada kwitansi uji, teks cetak terbaca rapi tapi isian
  tulisan tangan meleset jauh. Ini batas modelnya, bukan setelan.
- **Memori.** Di mesin 16 GB, tutup aplikasi berat lain saat memproses PDF besar.

## Rujukan

- Repo: https://github.com/baidu/Unlimited-OCR
- Bobot resmi: https://huggingface.co/baidu/Unlimited-OCR
- Paper: https://arxiv.org/abs/2606.23050
- Konversi MLX: https://huggingface.co/mlx-community/Unlimited-OCR-8bit
