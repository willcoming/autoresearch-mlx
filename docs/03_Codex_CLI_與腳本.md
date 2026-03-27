# 03｜Codex CLI 與腳本

如果你想用 CLI 而不是 app，這份給你。

---

## 1. 安裝 CLI
```bash
npm i -g @openai/codex
```

第一次執行：
```bash
codex
```

登入後就可以互動式使用。

---

## 2. 最推薦的 CLI 模式

### 互動式
```bash
codex
```

進 repo 後，直接丟類似這種 prompt：

```text
Read AGENTS.md, program.md, prepare.py, train.py, and README_zh-TW.md.
Do not change anything yet.
Summarize the objective, immutable files, baseline command, and the metric we optimize.
```

### 非互動式單次實驗
```bash
codex exec --full-auto "Read AGENTS.md and current train.py. Make exactly one contained experiment intended to improve val_sharpe. Edit train.py only. Run uv run train.py > run.log 2>&1. Report diff and metrics, then stop."
```

或直接吃 prompt 檔：
```bash
./scripts/codex_exec_once.sh prompts/02_單次安全實驗.txt
```

---

## 3. 這些腳本怎麼用

### `scripts/run_baseline.sh`
功能：
- `uv sync`
- 跑 baseline
- 把 log 存到 `logs/`

用法：
```bash
./scripts/run_baseline.sh
```

### `scripts/show_metrics.sh`
功能：
- 從 log 抓出關鍵指標

用法：
```bash
./scripts/show_metrics.sh run.log
```

### `scripts/append_result.py`
功能：
- 從 log 解析指標
- 追加到 `results.tsv`

用法：
```bash
python scripts/append_result.py run.log keep "baseline"
python scripts/append_result.py run.log discard "lookback 60 experiment"
```

### `scripts/codex_exec_once.sh`
功能：
- 用 `codex exec --full-auto` 跑一個 prompt 檔

用法：
```bash
./scripts/codex_exec_once.sh prompts/02_單次安全實驗.txt
```

---

## 4. 建議的 CLI 流程

### 第一步：先跑 baseline
```bash
./scripts/run_baseline.sh
python scripts/append_result.py run.log keep "baseline"
```

### 第二步：單次實驗
```bash
./scripts/codex_exec_once.sh prompts/02_單次安全實驗.txt
./scripts/show_metrics.sh run.log
python scripts/append_result.py run.log discard "single Codex experiment"
```

### 第三步：人工決定 keep / revert
你可以：

- 自己看 diff
- 或讓 Codex 先做 review
- 再決定要不要 commit

---

## 5. 為什麼我不建議一開始把 keep/revert 完全自動化
原因很簡單：

- 這是金融回測指標
- 很容易被噪音欺騙
- 你應該先看一下 diff 與風險，再決定是否保留

你可以半自動化：
- Codex 提案
- Codex 跑一次
- 你看報告
- 你決定 commit 還是 revert

這比完全放手更穩。

---

## 6. 如果你真的要進一步自動化
你可以之後再做：

- `codex exec`
- 外層 shell / Python wrapper
- 由 wrapper 讀 `run.log`
- 自動更新 `results.tsv`
- 再依規則決定保留或回退

但建議等你先跑順 10~20 次實驗，再做這步。
