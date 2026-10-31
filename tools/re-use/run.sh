#!/usr/bin/env bash
# 啟動 RE-USE demo（http://127.0.0.1:${PORT:-7860}）。先執行 setup.sh。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/.work}"
cd "$WORK/space"  # app.py 以相對路徑讀 recipes/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True exec "$WORK/.venv/bin/python" app.py
