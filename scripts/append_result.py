#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results.tsv"

METRIC_PATTERNS = {
    "val_sharpe": re.compile(r"^val_sharpe:\s+([-+0-9.eE]+)$", re.MULTILINE),
    "peak_vram_mb": re.compile(r"^peak_vram_mb:\s+([-+0-9.eE]+)$", re.MULTILINE),
    "locked_test_sharpe": re.compile(r"^locked_test_sharpe:\s+([-+0-9.eE]+)$", re.MULTILINE),
}

def git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "NA"
    except Exception:
        return "NA"

def extract_metric(text: str, key: str) -> str:
    m = METRIC_PATTERNS[key].search(text)
    return m.group(1) if m else ""

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/append_result.py <log_path> [status] [description]")
        return 1

    log_path = pathlib.Path(sys.argv[1]).resolve()
    status = sys.argv[2] if len(sys.argv) >= 3 else "keep"
    description = sys.argv[3] if len(sys.argv) >= 4 else ""

    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}")
        return 1

    text = log_path.read_text(encoding="utf-8", errors="replace")
    val_sharpe = extract_metric(text, "val_sharpe")
    peak_vram_mb = extract_metric(text, "peak_vram_mb")
    locked_test_sharpe = extract_metric(text, "locked_test_sharpe")

    if locked_test_sharpe and not val_sharpe:
        val_sharpe = f"locked:{locked_test_sharpe}"

    row = {
        "commit": git_commit_short(),
        "val_sharpe": val_sharpe,
        "peak_vram_mb": peak_vram_mb,
        "status": status,
        "description": description,
    }

    results_path = DEFAULT_RESULTS
    file_exists = results_path.exists()

    with results_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["commit", "val_sharpe", "peak_vram_mb", "status", "description"],
            delimiter="\t",
        )
        if not file_exists or results_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended row to {results_path}")
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
