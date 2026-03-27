# 02｜Codex app 操作與提示詞

這份文件專門講 **Codex app** 怎麼用在這個研究流程。

---

## 1. 先選對執行模式

### 建議：Local
這個模板是針對 `autoresearch-mlx` 改的，而 `autoresearch-mlx` 本身是 **MLX / Apple Silicon** 路線。  
所以真正執行 `uv run train.py` 時，最建議用：

- Codex app
- Local thread
- 你的 Apple Silicon 機器

### 不建議一開始就用 Cloud 跑訓練
Cloud 適合：
- 寫文件
- 看 diff
- 做 review
- 整理 changelog

但這個研究模板的訓練本身要吃本機 MLX 流程，Cloud 不是第一選擇。

---

## 2. 建議的 app 設定方式

### 第一次打開專案
1. 開 Codex app
2. 選你的 repo 資料夾
3. 確認是 Git repo
4. 選 **Local**
5. 開一個新 thread

### 啟動前先做的事
在 thread 第一則訊息，不要直接叫它改 code。先叫它確認規則：

> Read AGENTS.md, program.md, prepare.py, train.py, and README_zh-TW.md.  
> Do not change anything yet.  
> Summarize the objective, immutable files, baseline command, and the exact metric we optimize.

這樣能先確認它真的抓到規則。

---

## 3. 建議的 thread 節奏

### Thread A：環境檢查 + baseline
使用：
- `prompts/01_檢查環境與基線.txt`

目標：
- 先不改 code
- 確認 CSV 與依賴 OK
- 跑出第一個 baseline

### Thread B：單次安全實驗
使用：
- `prompts/02_單次安全實驗.txt`

目標：
- 只做一個小變動
- 只改 `train.py`
- 跑一次
- 回報 diff + metrics

### Thread C：失敗分析
使用：
- `prompts/03_分析失敗原因並提出下一步.txt`

目標：
- 不改 code
- 幫你排下一步優先級

---

## 4. 最好直接複製貼上的提示詞

### A. 第一次確認規則
請直接貼：
```text
Read AGENTS.md, program.md, prepare.py, train.py, and README_zh-TW.md.
Do not change anything yet.
Summarize:
1. the objective,
2. the immutable files,
3. the mutable file,
4. the baseline command,
5. the exact metric that decides keep vs revert.
Then stop.
```

### B. 跑 baseline
請直接貼：
```text
Read AGENTS.md and README_zh-TW.md.
Do not edit any files.
Run:
uv sync
uv run train.py > run.log 2>&1
Then report only:
- val_sharpe
- median_turnover
- median_drawdown
- fold_sharpes
- any setup problems
Stop.
```

### C. 單次小實驗
請直接貼：
```text
Read AGENTS.md and current train.py.
Make exactly one contained experiment intended to improve val_sharpe.
Rules:
- edit train.py only
- do not edit prepare.py, AGENTS.md, program.md, results.tsv, or data files
- do not install packages
- do not run multiple experiments
Before editing, state the hypothesis in 1-2 sentences.
After editing, run:
uv run train.py > run.log 2>&1
Then report:
- a concise diff summary
- val_sharpe
- median_turnover
- median_drawdown
- fold_sharpes
- keep or revert recommendation
Stop.
```

### D. 安全回退
請直接貼：
```text
If the latest experiment did not improve val_sharpe, revert only train.py to the last committed state, verify train.py is clean, and summarize the revert.
If the latest experiment improved val_sharpe, do not revert; just say so and stop.
```

---

## 5. 建議你怎麼 review

每次實驗結束後，看三件事：

### 1) `val_sharpe`
這是主指標。

### 2) `median_turnover`
如果 Sharpe 只多一點點，但 turnover 明顯更高，不一定值得留。

### 3) `median_drawdown`
避免 agent 透過極端風格拿到表面上比較好看的 Sharpe。

---

## 6. 建議的 keep / revert 準則

### keep
- `val_sharpe` 明顯更高
- 或 Sharpe 幾乎相同，但 turnover 更低 / drawdown 更小 / 程式更簡單

### revert
- `val_sharpe` 下降
- 變複雜但沒有帶來可驗證的收益
- 開始觸碰 `prepare.py` 或資料規則

---

## 7. 何時要用 worktree
如果你想平行測兩條路線，例如：

- 一條試 lookback / regularization
- 一條試 model architecture

請用 **worktree** 分開。  
不要讓兩個 thread 同時改同一個 checkout 的 `train.py`。

---

## 8. App 裡常用的小習慣

### composer 直接輸入 `/`
可以看目前可用的 slash commands。

### thread 太長
開新 thread，或先讓它整理目前狀態再換 thread。

### 每次只做一件事
不要在同一個 prompt 裡叫它：
- 改 architecture
- 改 optimizer
- 改 lookback
- 再順便幫你 commit

這樣很難知道是哪個改動有效。

---

## 9. 建議的 Local environments setup script
如果你常用 worktree，可以把下面內容放進 Codex app 的 Local environments setup script：

```bash
set -euo pipefail
mkdir -p logs data
uv sync
if [ ! -f data/GOOGL_1d_5y.csv ]; then
  echo "WARNING: data/GOOGL_1d_5y.csv is missing"
fi
```

同樣內容也已經放在：
- `scripts/codex_worktree_setup.sh`

---

## 10. 做完搜尋後
最後才跑：
```bash
REVEAL_FINAL_TEST=1 uv run train.py > final_test.log 2>&1
grep "^locked_test_" final_test.log
```

搜尋階段只盯 `val_sharpe`。
