#!/usr/bin/env bash
set -euo pipefail

LOG="${1:-run.log}"

if [ ! -f "$LOG" ]; then
  echo "ERROR: log file not found: $LOG"
  exit 1
fi

grep "^val_sharpe:\|^median_turnover:\|^median_drawdown:\|^median_cagr:\|^median_exposure:\|^fold_sharpes:\|^locked_test_" "$LOG" || true
