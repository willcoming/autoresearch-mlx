#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs data

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed."
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/baseline_${STAMP}.log"

uv sync
uv run train.py > "$LOG" 2>&1

cp "$LOG" run.log

echo "Wrote $LOG"
grep "^val_sharpe:\|^median_turnover:\|^median_drawdown:\|^fold_sharpes:" "$LOG" || true
