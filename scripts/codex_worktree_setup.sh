#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs data

if command -v uv >/dev/null 2>&1; then
  uv sync
else
  echo "WARNING: uv not found; install uv before running train.py"
fi

if [ ! -f data/GOOGL_1d_5y.csv ]; then
  echo "WARNING: data/GOOGL_1d_5y.csv is missing"
fi
