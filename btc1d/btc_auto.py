"""
btc_auto.py — BTC 日線自主策略優化迴圈
隨機擾動參數或訊號組合 → 回測 → 保留或還原。

使用方式:
    python btc_auto.py

按 Ctrl+C 停止。結果記錄在 btc1d_results.tsv。
主要指標：sharpe（越高越好）
"""

import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

STRATEGY_FILE = os.path.join(os.path.dirname(__file__), "btc_strategy.py")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "btc1d_results.tsv")
BACKUP_FILE = STRATEGY_FILE + ".bak"
LOG_FILE = os.path.join(os.path.dirname(__file__), "btc1d_run.log")

PARAM_SPACE = {
    "RSI_LENGTH": list(range(7, 43, 1)),
    "RSI_OVERSOLD": list(range(15, 46, 5)),
    "RSI_OVERBOUGHT": list(range(55, 86, 5)),
    "ZLSMA_LENGTH": list(range(10, 61, 5)),
    "MACD_FAST": list(range(5, 17, 1)),
    "MACD_SLOW": list(range(18, 36, 1)),
    "MACD_SIGNAL": list(range(5, 16, 1)),
}

ENTRY_COMBOS = [
    ["macd_bull", "rsi_exit_oversold", "price_breakout_zlsma"],
    ["macd_bull", "price_breakout_zlsma"],
    ["rsi_exit_oversold", "price_breakout_zlsma"],
    ["macd_bull", "rsi_exit_oversold"],
    ["ema5_cross_over_10", "macd_bull"],
    ["ema5_cross_over_10", "price_breakout_zlsma"],
    ["price_breakout_zlsma"],
    ["ema5_cross_over_10"],
    ["macd_bull"],
    ["consecutive2_day_up", "macd_bull"],
    ["consecutive2_day_up", "price_breakout_zlsma"],
]

EXIT_COMBOS = [
    ["macd_bear", "rsi_exit_overbought"],
    ["rsi_exit_overbought"],
    ["rsi_exit_overbought", "price_breakdown_zlsma"],
    ["rsi_exit_overbought", "consecutive2_day_down"],
    ["macd_bear"],
    ["price_breakdown_zlsma"],
    ["ema5_cross_under_10"],
    ["macd_bear", "price_breakdown_zlsma"],
    ["consecutive2_day_down"],
]


def read_current_params() -> dict:
    params = {}
    with open(STRATEGY_FILE) as f:
        for line in f:
            line = line.strip()
            for key in PARAM_SPACE:
                if line.startswith(key + " "):
                    m = re.search(r"=\s*(\d+)", line)
                    if m:
                        params[key] = int(m.group(1))
    return params


def read_current_combos() -> tuple:
    entry = exit_ = None
    with open(STRATEGY_FILE) as f:
        content = f.read()
    m = re.search(r"ENTRY_SIGNALS\s*=\s*(\[.*?\])", content, re.DOTALL)
    if m:
        entry = eval(m.group(1))
    m = re.search(r"EXIT_SIGNALS\s*=\s*(\[.*?\])", content, re.DOTALL)
    if m:
        exit_ = eval(m.group(1))
    return entry or [], exit_ or []


def write_param(key: str, value: int):
    with open(STRATEGY_FILE) as f:
        content = f.read()
    content = re.sub(
        rf"^({key}\s*=\s*)\d+",
        rf"\g<1>{value}",
        content,
        flags=re.MULTILINE,
    )
    with open(STRATEGY_FILE, "w") as f:
        f.write(content)


def write_combo(entry: list, exit_: list):
    with open(STRATEGY_FILE) as f:
        content = f.read()
    content = re.sub(
        r"ENTRY_SIGNALS\s*=\s*\[.*?\]",
        f"ENTRY_SIGNALS = {entry}",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"EXIT_SIGNALS\s*=\s*\[.*?\]",
        f"EXIT_SIGNALS = {exit_}",
        content,
        flags=re.DOTALL,
    )
    with open(STRATEGY_FILE, "w") as f:
        f.write(content)


def run_backtest() -> float | None:
    script = os.path.join(os.path.dirname(__file__), "btc_backtest.py")
    with open(LOG_FILE, "w") as log:
        proc = subprocess.run(
            [sys.executable, script],
            stdout=log,
            stderr=log,
            cwd=os.path.dirname(__file__),
        )

    if proc.returncode != 0:
        return None

    with open(LOG_FILE) as f:
        content = f.read()

    m = re.search(r"^sharpe:\s*([-\d.]+)", content, re.MULTILINE)
    if not m:
        return None

    return float(m.group(1))


def read_full_metrics() -> dict:
    metrics = {}
    if not os.path.exists(LOG_FILE):
        return metrics
    with open(LOG_FILE) as f:
        for line in f:
            for key in [
                "sharpe",
                "profit_factor",
                "win_rate",
                "max_drawdown_pct",
                "total_return_pct",
                "num_trades",
            ]:
                if line.startswith(key + ":"):
                    try:
                        metrics[key] = float(line.split(":")[1].strip().rstrip("%"))
                    except Exception:
                        pass
    return metrics


def init_results_tsv():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("timestamp\tsharpe\tprofit_factor\twin_rate\tmax_dd\ttotal_return\tnum_trades\tstatus\tdescription\n")


def log_result(metrics: dict, status: str, description: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = "\t".join([
        now,
        str(metrics.get("sharpe", "")),
        str(metrics.get("profit_factor", "")),
        str(metrics.get("win_rate", "")),
        str(metrics.get("max_drawdown_pct", "")),
        str(metrics.get("total_return_pct", "")),
        str(metrics.get("num_trades", "")),
        status,
        description,
    ])
    with open(RESULTS_FILE, "a") as f:
        f.write(row + "\n")


def main():
    print("=" * 60)
    print(" BTC 日線自主策略優化迴圈")
    print(" 按 Ctrl+C 停止")
    print("=" * 60)

    init_results_tsv()

    print("\n[初始化] 執行 baseline 回測...")
    shutil.copy(STRATEGY_FILE, BACKUP_FILE)
    baseline_sharpe = run_backtest()

    if baseline_sharpe is None:
        print("❌ Baseline 回測失敗，請確認數據已準備好（python3 btc1d/btc_prepare.py）")
        sys.exit(1)

    metrics = read_full_metrics()
    log_result(metrics, "keep", "baseline")
    best_sharpe = baseline_sharpe
    experiment_num = 1

    print(f"✓ Baseline sharpe = {best_sharpe:.4f}")
    print(
        f"  profit_factor={metrics.get('profit_factor', '?')}  "
        f"win_rate={metrics.get('win_rate', '?')}  "
        f"trades={metrics.get('num_trades', '?')}"
    )

    print("\n[開始自主優化迴圈]")

    while True:
        exp_start = time.time()
        params_before = read_current_params()
        entry_before, exit_before = read_current_combos()

        shutil.copy(STRATEGY_FILE, BACKUP_FILE)

        mutation_type = random.choices(["param", "combo"], weights=[70, 30])[0]

        if mutation_type == "param":
            key = random.choice(list(PARAM_SPACE.keys()))
            candidates = [v for v in PARAM_SPACE[key] if v != params_before.get(key)]
            if not candidates:
                continue
            new_val = random.choice(candidates)
            write_param(key, new_val)
            description = f"{key}: {params_before.get(key)}→{new_val}"
        else:
            new_entry = random.choice(ENTRY_COMBOS)
            new_exit = random.choice(EXIT_COMBOS)
            if new_entry == entry_before and new_exit == exit_before:
                new_entry = random.choice([c for c in ENTRY_COMBOS if c != entry_before])
            write_combo(new_entry, new_exit)
            description = f"ENTRY={new_entry} EXIT={new_exit}"

        print(f"\n[實驗 #{experiment_num}] {description}")
        new_sharpe = run_backtest()
        elapsed = time.time() - exp_start

        if new_sharpe is None:
            print(f"  ❌ 回測失敗（{elapsed:.0f}s），還原")
            shutil.copy(BACKUP_FILE, STRATEGY_FILE)
            log_result({}, "crash", description)
        elif new_sharpe > best_sharpe:
            metrics = read_full_metrics()
            improvement = new_sharpe - best_sharpe
            best_sharpe = new_sharpe
            log_result(metrics, "keep", description)
            print(f"  ✅ 改善！sharpe = {new_sharpe:.4f} (+{improvement:.4f})  [{elapsed:.0f}s]")
            print(
                f"     profit_factor={metrics.get('profit_factor', '?')}  "
                f"win_rate={metrics.get('win_rate', '?')}  "
                f"trades={metrics.get('num_trades', '?')}"
            )
        else:
            metrics = read_full_metrics()
            delta = new_sharpe - best_sharpe
            log_result(metrics, "discard", description)
            shutil.copy(BACKUP_FILE, STRATEGY_FILE)
            print(f"  ↩ 捨棄  sharpe = {new_sharpe:.4f} ({delta:+.4f})  [{elapsed:.0f}s]")

        experiment_num += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已停止。最終結果請查看 btc1d_results.tsv")
