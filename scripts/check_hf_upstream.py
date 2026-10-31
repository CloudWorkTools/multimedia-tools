"""比對 tools/*/upstream.json 鎖定的 Hugging Face commit 與上游目前的 commit，輸出 Markdown 報告。

有差異時，若在 GitHub Actions 中，寫入 drift=true 到 $GITHUB_OUTPUT。
用法：python scripts/check_hf_upstream.py [--self-test]
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = {"model": "models", "space": "spaces", "dataset": "datasets"}
WEB = {"model": "", "space": "spaces/", "dataset": "datasets/"}


def fetch_hf(kind, repo):
    with urllib.request.urlopen(f"https://huggingface.co/api/{API[kind]}/{repo}", timeout=30) as r:
        data = json.load(r)
    return data["sha"], data.get("lastModified", "")


def report(locks, fetch):
    rows, drift = [], False
    for tool, entries in locks:
        for name, e in entries.items():
            sha, modified = fetch(e["type"], e["repo"])
            changed = sha != e["sha"]
            drift |= changed
            url = f"https://huggingface.co/{WEB[e['type']]}{e['repo']}"
            rows.append(f"| {tool} | [{e['type']} `{e['repo']}`]({url}/commits/main) | `{e['sha'][:10]}` | `{sha[:10]}` | {modified[:10]} | "
                        + ("**有更新**" if changed else "相同") + " |")
    head = ["# 上游 Hugging Face 版本追蹤", "",
            "| 工具 | 上游 | 鎖定版本 | 上游最新 | 上游最後更新 | 狀態 |", "| --- | --- | --- | --- | --- | --- |"]
    tail = ["", "有更新時：檢視上游 commits，在本機以新 sha 執行 setup.sh 與 smoke_test.py，通過後更新 upstream.json。"
            if drift else "所有鎖定版本皆與上游相同。"]
    return "\n".join(head + rows + tail), drift


def self_test():
    locks = [("demo", {"model": {"repo": "a/b", "type": "model", "sha": "1" * 40},
                       "space": {"repo": "a/b", "type": "space", "sha": "2" * 40}})]
    same = lambda kind, repo: ({"model": "1", "space": "2"}[kind] * 40, "2026-01-01")
    moved = lambda kind, repo: ("3" * 40 if kind == "space" else "1" * 40, "2026-01-02")
    text, drift = report(locks, same)
    assert not drift and "所有鎖定版本皆與上游相同" in text, text
    text, drift = report(locks, moved)
    assert drift and text.count("**有更新**") == 1 and "spaces/a/b/commits/main" in text, text
    print("self-test ok")


def main():
    if "--self-test" in sys.argv:
        return self_test()
    locks = [(p.parent.name, json.loads(p.read_text())) for p in sorted(ROOT.glob("tools/*/upstream.json"))]
    text, drift = report(locks, fetch_hf)
    print(text)
    if out := os.environ.get("GITHUB_OUTPUT"):
        with open(out, "a") as f:
            f.write(f"drift={'true' if drift else 'false'}\n")


if __name__ == "__main__":
    main()
