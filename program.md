# autoresearch-mlx: trading mode for GOOGL daily bars

This protocol repurposes `autoresearch-mlx` from language-model tuning into a
fixed-data, fixed-backtester, autonomous policy-search loop for **GOOGL daily
bars**.

## Goal

Maximize `val_sharpe` on a fixed walk-forward validation schedule using only the
historical GOOGL daily bar file.

The system is **not** given a hand-written MA/RSI strategy. It only learns a
mapping:

- input: the last `N` daily bars as causal features
- output: next-day position weight `p_t`

## Immutable during the search loop

Treat these as fixed:

- `prepare.py`
- `data/GOOGL_1d_5y.csv`
- validation folds
- transaction cost model
- final test lock

Only `train.py` should be edited during the loop.

## Setup

1. Create a fresh branch such as `autoresearch/googl-sharpe-mar27`.
2. Copy the files from this package into the repo root:
   - `prepare.py`
   - `train.py`
   - `program.md`
   - `AGENTS.md`
   - `docs/`, `prompts/`, `scripts/`, `references/`
3. Put the market data CSV at `data/GOOGL_1d_5y.csv`.
4. Verify the CSV columns are exactly:
   - `date,open,high,low,close,volume`
5. Run one baseline:
   - `uv sync`
   - `uv run train.py > run.log 2>&1`
6. Record the baseline in `results.tsv`.

## What the agent may change

The agent may change only `train.py`, including:

- architecture
- hidden size / depth
- lookback length
- optimizer
- learning rate / weight decay
- regularization
- output mapping
- long-only vs long-short logic only if explicitly enabled in `prepare.py`

## What the agent must never do

- never modify `prepare.py`
- never modify the CSV or swap in another ticker
- never reveal or optimize on the final test set
- never use future information
- never shuffle time order
- never change transaction costs
- never install extra packages

## Selection metric

Primary metric:

- `val_sharpe` = median annualized out-of-sample Sharpe across the fixed
  walk-forward validation folds, after transaction costs

Tie-breakers:

1. lower turnover
2. lower drawdown
3. simpler code / fewer parameters

## Output format

The run should print lines like:

```text
---
val_sharpe:        0.842315
training_seconds:  78.3
peak_vram_mb:      512.4
median_turnover:   0.091203
median_drawdown:   0.143901
fold_sharpes:      fold_1=-0.1100, fold_2=0.4200, fold_3=1.0100, fold_4=0.8800
```

Useful commands:

```bash
grep "^val_sharpe:\|^median_turnover:\|^median_drawdown:" run.log
python scripts/append_result.py run.log keep "baseline or short description"
```

## Logging results

Use a tab-separated `results.tsv` with this header:

```text
commit	val_sharpe	peak_vram_mb	status	description
```

- `status` is one of `keep`, `discard`, or `crash`
- keep a run only if `val_sharpe` improved
- if `val_sharpe` is equal, prefer lower turnover / lower drawdown / simpler code

## Experiment loop

Recommended loop:

1. Start from the current kept commit.
2. Ask Codex to make exactly one bounded experiment.
3. Run `uv run train.py > run.log 2>&1`.
4. Read `val_sharpe`, turnover, and drawdown from `run.log`.
5. Record the result in `results.tsv`.
6. If improved, keep the change and commit it.
7. If worse, revert only `train.py`.

## Locked final test

The final test window is locked during search.

Only after the search is fully done may a human run:

```bash
REVEAL_FINAL_TEST=1 uv run train.py > final_test.log 2>&1
```

That path trains once on the full search-train window and then prints
`locked_test_*` metrics for the held-out final window. Do not use those numbers
for model selection during optimization.
