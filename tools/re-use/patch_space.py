"""把下載的 nvidia/RE-USE Space app.py 改成本機版。

上游程式碼不收錄在本 repo（NVIDIA 授權禁止散布），只在本機下載後套用這些修改。
每個錨點都必須剛好出現一次，否則上游已改版，直接失敗並列出缺少的錨點。

用法：python patch_space.py <space 目錄> <模型 revision>
"""
import sys
from pathlib import Path

CHUNK_RUN = '''        def run(wav):
            mag, pha, _ = mag_phase_stft(wav, n_fft=n_fft_scaled, hop_size=hop_size_scaled, win_size=win_size_scaled,
                                         compress_factor=compress_factor, center=True, addeps=False)
            amp_g, pha_g, _ = USE_model(mag, pha)
            # To remove "strange sweep artifact"
            m = torch.expm1(RELU(amp_g))
            zero_portion = torch.sum(m==0, 1)/m.shape[1]
            amp_g[:,:,(zero_portion>0.5)[0]] = 0
            g = mag_phase_istft(amp_g, pha_g, n_fft_scaled, hop_size_scaled, win_size_scaled, compress_factor)
            return pad_or_trim_to_match(wav.detach(), g, pad_value=1e-8)

        # 分段推論 + Hann 重疊相加（同 inference_chunk.py）：VRAM 峰值與片段長度成正比
        chunk = int(CHUNK_SECONDS * noisy_sr)
        if noisy_wav.shape[1] <= chunk:
            audio_g = run(noisy_wav)
        else:
            hop = chunk // 2
            window = torch.clamp(torch.hann_window(chunk).to(device), min=1e-7)
            audio_g = torch.zeros_like(noisy_wav)
            wsum = torch.zeros_like(noisy_wav)
            n = noisy_wav.shape[1]
            for s in list(range(0, n - chunk, hop)) + [n - chunk]:
                audio_g[:, s:s+chunk] += run(noisy_wav[:, s:s+chunk]) * window
                wsum[:, s:s+chunk] += window
            audio_g = audio_g / wsum
'''


def replace_once(src, old, new):
    count = src.count(old)
    if count != 1:
        raise LookupError(f"錨點出現 {count} 次（應為 1）：{old.strip()[:70]!r}")
    return src.replace(old, new)


def patch(src, revision):
    # 拿掉 ZeroGPU 的自動安裝區塊（本機已用 CUDA 編譯 mamba-ssm）
    start, end = src.find('CUDA_HOME = '), src.find('ABOUT = ')
    if start < 0 or end < start:
        raise LookupError("找不到安裝區塊（CUDA_HOME … ABOUT）")
    src = src[:start] + src[end:]

    # 換掉每次請求都重新載入模型的 stft→istft 區段
    start = src.find('        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(')
    end_anchor = '        assert audio_g.shape == noisy_wav.shape, audio_g.shape\n'
    end = src.find(end_anchor)
    if start < 0 or end < start:
        raise LookupError("找不到推論區段（mag_phase_stft … assert audio_g.shape）")
    src = src[:start] + CHUNK_RUN + src[end:]

    for old, new in [
        ('import spaces\n', ''),
        ('@spaces.GPU\n', ''),
        ('    USE_model = SEMamba.from_pretrained("nvidia/RE-USE", cfg=cfg1).to(device)\n    USE_model.eval()\n', ''),
        ('def enhance(',
         f'USE_model = SEMamba.from_pretrained("nvidia/RE-USE", cfg=cfg1, revision="{revision}").to(device).eval()\n\ndef enhance('),
        ('RELU = nn.ReLU()', 'RELU = nn.ReLU()\nCHUNK_SECONDS = float(os.environ.get("RE_USE_CHUNK_SECONDS", "4"))'),
        ('import matplotlib.pyplot as plt', 'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport librosa.display'),
        ('    return "original.wav", "enhanced.wav", fig',
         '    torch.cuda.empty_cache()  # 把快取還給驅動，nvidia-smi 才會反映實際用量\n    return "original.wav", "enhanced.wav", fig'),
        ('launch(share=True)', 'launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", "7860")), show_error=True)'),
    ]:
        src = replace_once(src, old, new)
    return src


def main():
    space, revision = Path(sys.argv[1]), sys.argv[2]
    app = space / "app.py"
    marker = "# patched-by: multimedia-tools/tools/re-use/patch_space.py\n"
    src = app.read_text()
    if src.startswith(marker):
        print("app.py 已套用過修改，略過")
        return
    try:
        out = patch(src, revision)
    except LookupError as e:
        sys.exit(f"上游 app.py 結構已變更，需更新 patch_space.py：{e}")
    app.write_text(marker + out)
    print(f"已修改 {app}")


if __name__ == "__main__":
    main()
