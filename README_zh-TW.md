# autoresearch-mlx / GOOGL Sharpe + Codex 打包說明

這個壓縮包不是獨立專案，而是 **覆蓋到 `trevin-creator/autoresearch-mlx` 的研究模板**。  
用途是把原本的 `autoresearch-mlx`，改造成：

- 標的：`GOOGL`
- 頻率：日 K
- 範圍：約 5 年
- 目標：最大化固定 walk-forward 驗證上的 **夏普率**
- 搜尋方式：讓 Codex 只改 `train.py`，自己找更好的 policy / 模型 / 超參數

---

## 1. 套件內容

### 核心覆蓋檔
- `prepare.py`  
  固定資料讀取、特徵處理、walk-forward 切分、交易成本、回測評分
- `train.py`  
  可被 agent 搜尋的主要檔案；預設是 MLX MLP baseline
- `program.md`  
  給 autoresearch / agent 的研究協議
- `AGENTS.md`  
  給 Codex 的持久化規則；Codex 啟動時會先讀它

### 使用說明
- `docs/01_快速開始.md`
- `docs/02_Codex_App_操作與提示詞.md`
- `docs/03_Codex_CLI_與腳本.md`
- `docs/04_常見錯誤與排查.md`
- `docs/05_操作流程速查.md`

### 可直接複製的提示詞
- `prompts/01_檢查環境與基線.txt`
- `prompts/02_單次安全實驗.txt`
- `prompts/03_分析失敗原因並提出下一步.txt`
- `prompts/04_整理目前最佳版本.txt`
- `prompts/05_安全回退.txt`
- `prompts/06_最終鎖定測試前檢查.txt`
- `prompts/07_執行最終鎖定測試.txt`

### 小工具
- `scripts/run_baseline.sh`
- `scripts/show_metrics.sh`
- `scripts/append_result.py`
- `scripts/codex_exec_once.sh`
- `scripts/codex_worktree_setup.sh`

### 其他
- `data/GOOGL_1d_5y.template.csv`
- `results.tsv`
- `references/official_links.md`

---

## 2. 最推薦的使用方式

### 最推薦：Codex app + Local thread
這個專案依賴 **MLX / Apple Silicon**，所以最穩的方式是：

1. 在 Apple Silicon Mac 上開 repo
2. 用 Codex app 選 **Local**
3. 讓 Codex 在你的本機工作目錄裡改 `train.py`
4. 每次只做一個小實驗
5. 看 `run.log` 的 `val_sharpe`

> 不建議一開始就用 Cloud thread 跑這個研究 loop，因為這個 repo 是 MLX/Apple Silicon 取向；Cloud 適合做 code review、整理 diff、寫文件，但不一定適合直接跑你這個本機 MLX 訓練流程。

---

## 3. 快速開始

### Step 1：先取得上游 repo
```bash
git clone https://github.com/trevin-creator/autoresearch-mlx.git
cd autoresearch-mlx
git checkout -b autoresearch/googl-sharpe-mar27
```

### Step 2：備份原始檔
至少先備份：
- `prepare.py`
- `train.py`
- `program.md`

例如：
```bash
cp prepare.py prepare.py.bak
cp train.py train.py.bak
cp program.md program.md.bak
```

### Step 3：把這個壓縮包內容複製到 repo root
把下列檔案／資料夾複製到 repo 根目錄：
- `prepare.py`
- `train.py`
- `program.md`
- `AGENTS.md`
- `docs/`
- `prompts/`
- `scripts/`
- `references/`
- `results.tsv`

### Step 4：放入 GOOGL 日 K CSV
把你的 5 年日 K 檔放到：
```text
data/GOOGL_1d_5y.csv
```

欄位必須是：
```csv
date,open,high,low,close,volume
```

### Step 5：同步依賴
```bash
uv sync
```

### Step 6：先跑 baseline
```bash
uv run train.py > run.log 2>&1
./scripts/show_metrics.sh run.log
python scripts/append_result.py run.log keep "baseline"
```

---

## 4. 資料檔要求

### 必要欄位
CSV header 必須是：
```csv
date,open,high,low,close,volume
```

### 建議格式
- `date`：`YYYY-MM-DD`
- 排序：由舊到新
- 不要混入調整前後不同口徑的價格
- 不要把別的 ticker 混進同一檔

### 注意
`prepare.py` 會自己做基本檢查，包含：
- 必要欄位是否存在
- 樣本數是否夠長
- 特徵與 target 對齊是否合理

---

## 5. 這個研究模板的設計原則

### 你沒有把策略硬塞給 agent
這份模板沒有寫死：
- MA crossover
- RSI entry/exit
- MACD rule
- breakout rule

固定的是：

- 資料
- 交易成本
- 回測規則
- 驗證切分
- 夏普評分器

可變的是：

- 模型結構
- lookback 長度
- 正則化
- optimizer
- 倉位映射方式
- 其他只發生在 `train.py` 裡的學習邏輯

也就是說，agent 搜尋的是：
**「過去 N 天日 K 特徵 → 明天倉位」**  
而不是你手寫的傳統規則策略。

---

## 6. 如何在 Codex app 裡操作

請先看：
- `docs/02_Codex_App_操作與提示詞.md`
- `prompts/` 目錄裡的複製貼上提示詞

最短版操作流程：

1. 打開 Codex app
2. 選 repo 專案資料夾
3. 選 **Local**
4. 先丟：
   - `prompts/01_檢查環境與基線.txt`
5. baseline 沒問題後，再丟：
   - `prompts/02_單次安全實驗.txt`
6. 每次只讓它做 **一個** 實驗
7. 看 diff 與 `run.log`
8. 好的保留，不好的回退

---

## 7. 如果你想用 Codex CLI

請看：
- `docs/03_Codex_CLI_與腳本.md`

最短版：

```bash
npm i -g @openai/codex
codex
```

或單次非互動：

```bash
./scripts/codex_exec_once.sh prompts/02_單次安全實驗.txt
```

---

## 8. 什麼時候才看最終鎖定測試

搜尋期間只看 `val_sharpe`。  
**不要**在搜尋中用 locked final test 做選模。

只有當你確定目前版本要收斂時，才跑：

```bash
REVEAL_FINAL_TEST=1 uv run train.py > final_test.log 2>&1
grep "^locked_test_" final_test.log
```

---

## 9. 建議工作習慣

- 一次只做一個改動
- 一個 thread 對應一個清楚的小目標
- 不要讓兩個 thread 同時改同一個 checkout 的 `train.py`
- 如果要平行搜尋，請用 Codex app 的 worktrees
- 每次實驗都記到 `results.tsv`
- thread 太長就開新 thread 或做 compact

---

## 10. 先看哪幾份文件

如果你只想最快上手，依序看：

1. `docs/01_快速開始.md`
2. `docs/02_Codex_App_操作與提示詞.md`
3. `docs/05_操作流程速查.md`
