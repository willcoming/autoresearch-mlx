#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROMPT_FILE="${1:-prompts/02_單次安全實驗.txt}"

if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex CLI is not installed."
  exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
  echo "ERROR: prompt file not found: $PROMPT_FILE"
  exit 1
fi

codex exec --full-auto "$(cat "$PROMPT_FILE")"
