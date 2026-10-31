# 多媒體工具

CloudWorkTools 的多媒體工具集中在這個 repo，每個工具一個目錄，位於 `tools/` 之下。Agent Skills 另放在 [multimedia-communication-skills](https://github.com/CloudWorkTools/multimedia-communication-skills)。

| 工具 | 用途 | 上游 |
| --- | --- | --- |
| [re-use](tools/re-use/README.md) | 本機語音增強 demo：上傳含雜訊的音檔或影片，輸出增強後的語音與前後頻譜圖。 | [nvidia/RE-USE](https://huggingface.co/nvidia/RE-USE) |

## 上游追蹤

[`sync-upstream.yml`](.github/workflows/sync-upstream.yml) 每天 01:37 與 13:37（UTC）比對各工具 `upstream.json` 鎖定的 Hugging Face commit 與上游目前的 commit。有差異時，更新標題為「[追蹤] Hugging Face 上游有更新」的 issue；沒有差異時只寫入該次執行的摘要。

上游更新不會自動套用。依 issue 中的連結檢視上游 commits，在本機以新的 sha 重跑該工具的安裝與冒煙測試，通過後再更新 `upstream.json`。

本機手動比對：

```bash
python3 scripts/check_hf_upstream.py
```

## 新增工具

1. 建立 `tools/<名稱>/`，放入 `README.md`，說明用途、需求、安裝與執行方式。
2. 工具依賴 Hugging Face 上的模型、Space 或資料集時，加入 `upstream.json`，追蹤 workflow 會自動納入：

   ```json
   {"model": {"repo": "<owner>/<name>", "type": "model", "sha": "<40 字元 commit>"}}
   ```

   `type` 可為 `model`、`space` 或 `dataset`；鍵名（如 `model`）只用於報告顯示。
3. 在上方表格加一列。

上游授權禁止散布的程式碼不收進 repo：只收本 repo 自己的腳本，在本機下載上游後再套用修改，做法見 [re-use](tools/re-use/README.md)。

## 授權

本 repo 的程式碼採 [MIT](LICENSE)。各工具下載的上游模型與程式碼適用其各自的授權，見各工具的 README。
