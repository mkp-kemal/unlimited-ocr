#!/usr/bin/env bash
# Nyalakan demo OCR beserta server yang dibutuhkannya.
#
# Dua proses harus hidup, dan keduanya mati setiap laptop direstart:
#   - app Gradio di :7860
#   - server mlx-vlm di :8111, dipakai engine PaddleOCR-VL
#
#   ./jalankan.sh          nyalakan yang belum hidup
#   ./jalankan.sh stop     matikan keduanya
set -euo pipefail
cd "$(dirname "$0")"

hidup() { lsof -ti tcp:"$1" >/dev/null 2>&1; }

tunggu() {  # tunggu port naik, maksimal 60 detik
    for _ in $(seq 60); do
        hidup "$1" && return 0
        sleep 1
    done
    return 1
}

if [ "${1:-}" = "stop" ]; then
    for port in 7860 8111; do
        if hidup "$port"; then
            kill "$(lsof -ti tcp:"$port")" && echo "port $port dimatikan"
        else
            echo "port $port memang tidak jalan"
        fi
    done
    exit 0
fi

if hidup 8111; then
    echo "server mlx-vlm :8111 sudah jalan"
else
    echo "menyalakan server mlx-vlm :8111 (muat model ~20 detik)..."
    nohup .venv/bin/python -m mlx_vlm.server --port 8111 \
        --model PaddlePaddle/PaddleOCR-VL-1.6 --trust-remote-code \
        > /tmp/mlx-server.log 2>&1 &
    tunggu 8111 && echo "  siap" || { echo "  GAGAL — lihat /tmp/mlx-server.log"; exit 1; }
fi

if hidup 7860; then
    echo "app :7860 sudah jalan"
else
    echo "menyalakan app :7860..."
    nohup .venv/bin/python app.py > /tmp/gradio-ocr.log 2>&1 &
    tunggu 7860 && echo "  siap" || { echo "  GAGAL — lihat /tmp/gradio-ocr.log"; exit 1; }
fi

echo
echo "Buka http://127.0.0.1:7860"
