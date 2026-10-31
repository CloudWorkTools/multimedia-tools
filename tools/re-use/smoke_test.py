"""對執行中的 demo 送一段音檔，回報耗時與整張 GPU 的 VRAM 峰值；失敗時以非零值結束。

用法：python smoke_test.py <音檔> [--url http://127.0.0.1:7860]
需要 gradio_client（setup.sh 建的 venv 已含）。
"""
import argparse
import subprocess
import threading
import time

from gradio_client import Client, handle_file


def gpu_used_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(out.split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    args = ap.parse_args()

    before, peak, done = gpu_used_mib(), [0], threading.Event()

    def sample():
        while not done.is_set():
            peak[0] = max(peak[0], gpu_used_mib())
            time.sleep(0.3)

    threading.Thread(target=sample, daemon=True).start()
    t = time.time()
    _, enhanced, _ = Client(args.url, verbose=False).predict(
        handle_file(args.audio), None, "", "", api_name="/unified_enhance")
    elapsed = time.time() - t
    done.set()
    time.sleep(0.4)
    print(f"輸出：{enhanced}")
    print(f"耗時 {elapsed:.1f}s；整張 GPU 峰值 {peak[0]} MiB（請求前 {before} MiB，含其他程序）")


if __name__ == "__main__":
    main()
