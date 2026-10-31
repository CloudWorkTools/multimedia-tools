# RE-USE 本機語音增強 demo

在本機 GPU 執行 [nvidia/RE-USE](https://huggingface.co/nvidia/RE-USE) 的 Gradio demo（`http://127.0.0.1:7860`）：上傳含雜訊的音檔或影片，輸出增強後的語音與前後頻譜圖。介面來自官方 [Space](https://huggingface.co/spaces/nvidia/RE-USE)，改為本機執行並分段推論。

## 授權

模型採 [NVIDIA One-Way Noncommercial License](https://github.com/NVlabs/HMAR/blob/main/LICENSE)，僅限研究與開發，不得商用。Space 的 `app.py` 標註未經 NVIDIA 授權禁止散布，因此本目錄不收錄它：`setup.sh` 在本機下載鎖定版本的 Space，再由 `patch_space.py` 套用本機修改。

## 需求

- NVIDIA GPU 與驅動（已驗證：RTX 4060 Ti 16GB，驅動 595.99）。
- CUDA Toolkit 的 `nvcc`，主版本須與 PyTorch 相同（PyTorch 2.15.0 為 cu132；已驗證 nvcc 13.3）。
- [uv](https://docs.astral.sh/uv/)、[hf CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)、FFmpeg。

Ubuntu 26.04 安裝 nvcc 13.3（需要 sudo，下載數 GB）：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-3
/usr/local/cuda-13.3/bin/nvcc --version   # 應顯示 release 13.3
```

## 安裝與執行

```bash
./setup.sh   # 首次約需數分鐘編譯 CUDA 擴充；可重複執行
./run.sh     # 開啟 http://127.0.0.1:7860
```

所有產物放在 `.work/`（已排除於 git）。編譯失敗時見 [BUILD.md](BUILD.md)。

| 環境變數 | 預設 | 用途 |
| --- | --- | --- |
| `RE_USE_CHUNK_SECONDS` | `4` | 分段推論的片段長度（秒），決定 VRAM 峰值，見下方。 |
| `PORT` | `7860` | 服務埠。 |
| `CUDA_HOME` | `/usr/local/cuda-13.3` | `setup.sh` 使用的 CUDA Toolkit。 |
| `TORCH_CUDA_ARCH_LIST` | 偵測到的 GPU 架構 | `setup.sh` 編譯的目標架構。 |
| `WORK` | `tools/re-use/.work` | 下載、venv 與編譯產物的位置。 |

## 冒煙測試

服務啟動後，送一段音檔並回報耗時與 VRAM 峰值：

```bash
.work/.venv/bin/python smoke_test.py <音檔.wav>
```

## VRAM 與片段長度

模型權重只占 37MB。VRAM 峰值來自推論時的中間張量，與片段長度和取樣率成正比；音檔超過一個片段時，以 Hann 窗重疊相加接回。以下為 fp32 單次前向的峰值（`torch.cuda.max_memory_allocated`，RTX 4060 Ti）：

| 取樣率 | 2 秒 | 4 秒 | 8 秒 |
| --- | --- | --- | --- |
| 44.1kHz | 1.03 GiB | 2.05 GiB | 3.76 GiB |
| 16kHz | 0.38 GiB | 0.69 GiB | 1.37 GiB |

44.1kHz 下 30 秒的片段會在卷積層 OOM，所以長音檔必須分段。片段越短，重疊區的重複運算與接縫越多；不同片段長度對音質的影響尚未比較。

在預設的 4 秒片段下，25 秒、44.1kHz 的音檔約需 13 秒處理完；啟動後的第一次請求含暖機，約 24 秒。

## 與官方 Space 的差異

`patch_space.py` 對下載的 `app.py` 做以下修改；上游結構改變而對不上時，它會失敗並列出缺少的錨點。

- 移除 ZeroGPU（`spaces`）與啟動時自動安裝 mamba-ssm 的步驟。
- 模型在啟動時載入一次，並鎖定 `upstream.json` 中的 revision；官方版每次請求重新載入且追蹤 `main`。
- 推論改為分段重疊相加。
- 只聽 `127.0.0.1`，開啟 `show_error` 以在介面顯示錯誤。
- 每次請求後釋放 CUDA 快取。

## 已知限制

- 影片上傳與頻寬延伸選項沒有在本機測試過。
- 增強後的音質沒有做過聽測或客觀指標比較。
