# 編譯 mamba-ssm 的陷阱

`setup.sh` 已處理以下每一項。這份文件說明原因，供修改 `setup.sh` 或排查失敗時參考。

## 沒有 nvcc 時只能用純 PyTorch 版本

mamba-ssm 的 selective scan 有 CUDA kernel 與純 PyTorch 參考實作（`selective_scan_ref`）兩種。驅動不含編譯器，沒有 `nvcc` 就只能設 `MAMBA_SKIP_CUDA_BUILD=TRUE` 安裝，此時 `mamba_ssm` 匯入時仍會 `import selective_scan_cuda` 而失敗。以 4 秒片段處理 25 秒、44.1kHz 的音檔，純 PyTorch 版本需要 5 分 12 秒，CUDA kernel 版本約 15 秒，所以 `setup.sh` 要求 `nvcc`。

## PyTorch 2.15 需要 C++20

PyTorch 2.15 的標頭檔要求 C++20（`ATen.h`：`#error C++20 or later compatible compiler is required`），但 causal-conv1d 1.7.0 與 mamba-ssm 2.3.2.post1 的 `setup.py` 寫死 `-std=c++17`。`setup.sh` 從 PyPI 下載這兩個版本的原始碼，把旗標改成 `-std=c++20` 後再編譯。旗標格式若在新版改變，`setup.sh` 會停止。

## nvcc 的主版本要與 PyTorch 相同

PyTorch 2.15.0 以 CUDA 13.2 建置。Ubuntu 套件庫的 `nvidia-cuda-toolkit` 是 12.4，主版本不符；NVIDIA 的 `ubuntu2604` 套件庫沒有 13.2，只有 13.3 與 13.4。實測 13.3 可以編譯並正確匯入。

## 「安裝成功」不代表有 CUDA 擴充

uv 會重用快取中先前以 `SKIP_CUDA_BUILD` 建出的 wheel，幾毫秒就回報安裝成功，但沒有任何 `.so`。`setup.sh` 因此用 `--no-cache` 編譯，並以實際匯入判定成功：

```bash
python -c "import torch, selective_scan_cuda, causal_conv1d_cuda"
```

必須先 `import torch`，否則會因找不到 `libc10.so` 失敗，那是測試方式錯誤，不是編譯失敗。

## 同時只能有一輪編譯

兩輪編譯同時進行時，會寫入同一批暫存檔，失敗訊息也會混在一起，無法判斷是哪一輪的錯誤。另外 bash 是邊執行邊讀取腳本檔，執行中覆寫腳本會讓正在跑的那一輪讀到錯亂的指令。`setup.sh` 在偵測到 `ninja` 或 `nvcc` 執行中時拒絕開始，且每輪寫入以時間命名的新 log。
