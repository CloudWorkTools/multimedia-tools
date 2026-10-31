#!/usr/bin/env bash
# 建立 RE-USE 本機 demo：下載鎖定版本的 Space、套用本機修改、建 venv、以 CUDA 編譯 mamba-ssm。
# 可重複執行；已完成的步驟會略過。背景與疑難排解見 BUILD.md。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/.work}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
ARCH="${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)}"
CAUSAL_CONV1D=1.7.0
MAMBA_SSM=2.3.2.post1

[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "找不到 $CUDA_HOME/bin/nvcc，請先依 README 安裝 CUDA Toolkit" >&2; exit 1; }
command -v uv >/dev/null || { echo "需要 uv：curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
command -v hf >/dev/null || { echo "需要 hf CLI：curl -LsSf https://hf.co/cli/install.sh | bash" >&2; exit 1; }
mkdir -p "$WORK"

space_sha=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['space']['sha'])" "$HERE/upstream.json")
model_sha=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['model']['sha'])" "$HERE/upstream.json")

echo "== 下載 Space @ $space_sha"
hf download nvidia/RE-USE --repo-type space --revision "$space_sha" --local-dir "$WORK/space" >/dev/null
python3 "$HERE/patch_space.py" "$WORK/space" "$model_sha"

echo "== 建立 venv"
[ -x "$WORK/.venv/bin/python" ] || uv venv -q -p 3.11 "$WORK/.venv"
VIRTUAL_ENV="$WORK/.venv" uv pip install -q -r "$HERE/requirements.txt"
PY="$WORK/.venv/bin/python"

if "$PY" -c "import torch, selective_scan_cuda, causal_conv1d_cuda" 2>/dev/null; then
  echo "== CUDA 擴充已存在，略過編譯"
else
  # 另一輪編譯還在跑時重建會互相覆寫 log 與暫存檔
  if pgrep -x ninja >/dev/null || pgrep -x nvcc >/dev/null; then
    echo "已有 ninja/nvcc 在執行，請等它結束或先停止它" >&2; exit 1
  fi
  echo "== 下載並修改 causal-conv1d $CAUSAL_CONV1D / mamba-ssm $MAMBA_SSM 原始碼（C++17 → C++20）"
  rm -rf "$WORK/src" && mkdir -p "$WORK/src"
  for spec in "causal-conv1d $CAUSAL_CONV1D" "mamba-ssm $MAMBA_SSM"; do
    set -- $spec
    url=$(curl -fsSL "https://pypi.org/pypi/$1/$2/json" | python3 -c "import sys,json;print(next(f['url'] for f in json.load(sys.stdin)['urls'] if f['packagetype']=='sdist'))")
    curl -fsSL "$url" | tar xz -C "$WORK/src"
  done
  sed -i 's/-std=c++17/-std=c++20/' "$WORK"/src/*/setup.py
  if grep -q -- '-std=c++17' "$WORK"/src/*/setup.py || ! grep -q -- '-std=c++20' "$WORK"/src/*/setup.py; then
    echo "setup.py 的 -std 旗標格式已變，請檢查 $WORK/src/*/setup.py" >&2; exit 1
  fi

  log="$WORK/build-$(date +%Y%m%d-%H%M%S).log"
  echo "== 以 nvcc 編譯（sm_${ARCH/./}），log：$log"
  PATH="$CUDA_HOME/bin:$PATH" CUDA_HOME="$CUDA_HOME" TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-4}" \
  MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE VIRTUAL_ENV="$WORK/.venv" \
    uv pip install --no-cache --no-build-isolation --no-deps \
      "$WORK/src/causal_conv1d-$CAUSAL_CONV1D" "$WORK/src/mamba_ssm-$MAMBA_SSM" > "$log" 2>&1 \
    || { echo "編譯失敗，見 $log" >&2; exit 1; }
  # 安裝成功不代表有 CUDA 擴充（快取的 skip-build wheel 也會「成功」），以實際 import 為準
  "$PY" -c "import torch, selective_scan_cuda, causal_conv1d_cuda" \
    || { echo "編譯完成但無法匯入 CUDA 擴充，見 $log" >&2; exit 1; }
fi

echo "== 完成。啟動：$HERE/run.sh"
