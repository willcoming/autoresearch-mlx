#!/usr/bin/env python3
"""
Phase 2 only — uses pre-selected top configs from Run-26 exploration.
Skips Phase 1 entirely; should complete in ~10-15 minutes.
"""
import re
import sys
sys.path.insert(0, "/home/user/autoresearch-mlx")

from stock_pattern_research import (
    FeatureConfig, phase2_transfer_wf, walk_forward_backtest,
    fetch_stock_data, TEST_SYMBOLS, TRAIN_SYMBOL,
    INITIAL_TRAIN_DAYS, STEP_DAYS,
)
import numpy as np
from datetime import datetime


def parse_cfg(s: str) -> FeatureConfig:
    """Parse a config string like 'w[5, 10, 20]|lb20|macd|bb|c0.60'."""
    parts = s.split("|")
    # windows
    win_str = parts[0]  # e.g. "w[5, 10, 20]"
    windows = list(map(int, re.findall(r"\d+", win_str)))
    lb = 10
    conf = 0.50
    vol = rsi = macd = bb = atr = trend = stoch = regime = False
    for p in parts[1:]:
        if p.startswith("lb"):
            lb = int(p[2:])
        elif p == "vol":   vol    = True
        elif p == "rsi":   rsi    = True
        elif p == "macd":  macd   = True
        elif p == "bb":    bb     = True
        elif p == "atr":   atr    = True
        elif p == "tr":    trend  = True
        elif p == "stoch": stoch  = True
        elif p == "reg":   regime = True
        elif p.startswith("c"):
            conf = float(p[1:])
    return FeatureConfig(windows, vol, rsi, macd, bb, atr, trend,
                         stoch, regime, lookback=lb, conf_threshold=conf)


# ── Top-20 configs from Run-26 Phase 1 (n_trades >= 5, ranked by sharpe) ──
TOP20_RAW = [
    ("w[5, 10, 20]|lb20|macd|bb",          "lr", 2.881),
    ("w[5, 10, 20]|lb20|macd|tr|c0.55",    "lr", 2.754),
    ("w[5, 10, 20]|lb20|macd|c0.55",       "lr", 2.740),
    ("w[5, 10, 20]|lb14|rsi|atr|c0.60",    "lr", 2.738),
    ("w[5, 10, 20]|lb20|macd|tr|c0.60",    "lr", 2.725),
    ("w[5, 10, 20]|lb20|macd|c0.60",       "lr", 2.626),
    ("w[5, 10, 20]|lb20|macd",             "lr", 2.592),
    ("w[5, 10, 20]|lb14|rsi|reg",          "lr", 2.577),
    ("w[5, 10, 20]|lb20|macd|reg",         "lr", 2.484),
    ("w[5, 10, 20]|lb20|macd|c0.65",       "lr", 2.259),
    ("w[5, 10, 20]|lb14|tr",               "lr", 2.259),
    ("w[5, 10, 20]|lb20|reg",              "lr", 2.244),
    ("w[5, 10, 20]|lb20|rsi|macd|atr|tr",  "lr", 2.219),
    ("w[5, 10, 20]|lb14|rsi|atr|reg",      "lr", 2.184),
    ("w[5, 10, 20]|lb20|bb",               "lr", 2.132),
    ("w[5, 10, 20]|lb14|rsi|atr",          "lr", 2.114),
    ("w[5, 10, 20]|lb20|macd|tr",          "lr", 2.105),
    ("w[5, 10, 20]|lb14|rsi|atr|c0.55",    "lr", 2.100),
    ("w[5, 10, 20]|lb14|rsi",              "lr", 2.022),
    ("w[5, 10, 20]|lb20|atr|tr",           "lr", 2.000),
]

# Reconstruct explore_results as list-of-dicts (w_score used for ranking)
explore_results = []
for cfg_str, mdl, sharpe in TOP20_RAW:
    cfg = parse_cfg(cfg_str)
    explore_results.append({
        "cfg":          cfg,
        "model":        mdl,
        "w_score":      sharpe,   # use Phase-1 sharpe as w_score proxy
        "sharpe":       sharpe,
        "n_trades":     20,       # placeholder ≥ 5
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "win_rate":     0.60,
        "f1":           0.20,
        "peak_f1":      0.15,
        "valley_f1":    0.25,
    })

best_cfg   = parse_cfg("w[5, 10, 20]|lb20|macd|bb")
best_model = "lr"

print("╔══════════════════════════════════════════════════════════════════╗")
print("║   Phase-2 Only Run  (using Run-26 Phase-1 configs)              ║")
print(f"║   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  train={TRAIN_SYMBOL}  |  test={len(TEST_SYMBOLS)} stocks          ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print(f"\n  Top config : {best_cfg}  [{best_model}]")
print(f"  Candidate pool : {len(explore_results)} configs loaded from Run-26\n")

import time
t0 = time.time()

transfer_results = phase2_transfer_wf(
    best_cfg, best_model, TEST_SYMBOLS,
    explore_results=explore_results,
)

valid = [r for r in transfer_results if "error" not in r]
if valid:
    avg_sh  = sum(r["sharpe"]       for r in valid) / len(valid)
    avg_ret = sum(r["total_return"] for r in valid) / len(valid)
    best_s  = max(valid, key=lambda r: r["sharpe"])
    worst_s = min(valid, key=lambda r: r["sharpe"])
    print(f"\n{'═'*68}")
    print("  FINAL SUMMARY  (Walk-Forward — full 4 years OOS)")
    print(f"{'═'*68}")
    print(f"  ── Transfer-test ({len(valid)} stocks, 4yr walk-forward) ──")
    print(f"  Avg Sharpe       : {avg_sh:.3f}")
    print(f"  Avg return       : {avg_ret*100:.1f}%")
    print(f"  Best  stock      : {best_s['symbol']}  (Sharpe={best_s['sharpe']:.2f})")
    print(f"  Worst stock      : {worst_s['symbol']}  (Sharpe={worst_s['sharpe']:.2f})")
    print(f"  Total time       : {time.time()-t0:.1f}s")
