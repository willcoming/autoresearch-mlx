# GOOGL Sharpe search instructions for Codex

## Objective
Maximize `val_sharpe`, defined as the median annualized out-of-sample Sharpe
ratio across the fixed walk-forward validation folds for `GOOGL` daily bars.

## Immutable
Treat these as fixed during the search loop:

- `prepare.py`
- `data/GOOGL_1d_5y.csv`
- fold construction and time order
- transaction cost assumptions
- the locked final test split

## Mutable
Only `train.py` may be edited during the search loop.

## Required workflow for each experiment
1. Read `README_zh-TW.md`, `program.md`, `prepare.py`, and `train.py`.
2. Propose exactly one contained experiment.
3. Edit `train.py` only.
4. Run `uv run train.py > run.log 2>&1`.
5. Summarize:
   - the hypothesis,
   - the diff,
   - `val_sharpe`,
   - `median_turnover`,
   - `median_drawdown`,
   - `fold_sharpes`,
   - whether the change should be kept or reverted.
6. Stop after one experiment. Do not chain multiple experiments in one turn.

## Never do these
- never edit `prepare.py`
- never edit the CSV or change the ticker
- never install new packages
- never fetch live market data during the search loop
- never use future information or shuffle time order
- never optimize on the locked final test
- never run two threads that modify the same checkout; use a worktree if you want parallel experiments

## Selection rule
Primary metric: higher `val_sharpe`.

Tie-breakers, in order:
1. lower turnover
2. lower drawdown
3. simpler code and fewer parameters

## Useful commands
- Baseline run:
  `uv sync && uv run train.py > run.log 2>&1`
- Show summary metrics:
  `grep "^val_sharpe:\|^median_turnover:\|^median_drawdown:\|^fold_sharpes:" run.log`
- Reveal locked final test only after the search is over:
  `REVEAL_FINAL_TEST=1 uv run train.py > final_test.log 2>&1`

## Output discipline
Keep the response concise and operational. Prefer bullet lists with concrete
numbers over long explanations.
