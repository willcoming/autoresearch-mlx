"""
btc_backtest.py — 固定不修改
執行單次回測，讀取 btc_strategy.py 的參數與邏輯，輸出績效摘要。

使用方式:
    python btc_backtest.py

輸出格式（供 btc_auto.py grep 解析）:
    sharpe: 1.8500
    profit_factor: 1.9200
    ...
"""

import importlib
import sys
import os

# 確保可以 import 同目錄的模組
sys.path.insert(0, os.path.dirname(__file__))

from btc_prepare import load_btc_data, compute_indicators, run_backtest, print_summary


def main():
    # 動態載入策略（支援 btc_auto.py 修改後重新載入）
    import btc_strategy as s
    importlib.reload(s)

    params = {
        "RSI_LENGTH":    s.RSI_LENGTH,
        "RSI_OVERSOLD":  s.RSI_OVERSOLD,
        "RSI_OVERBOUGHT": s.RSI_OVERBOUGHT,
        "ZLSMA_LENGTH":  s.ZLSMA_LENGTH,
        "MACD_FAST":     s.MACD_FAST,
        "MACD_SLOW":     s.MACD_SLOW,
        "MACD_SIGNAL":   s.MACD_SIGNAL,
        "COMMISSION":    s.COMMISSION,
    }

    print("載入數據...")
    df = load_btc_data()

    print("計算指標...")
    df = compute_indicators(df, params)

    print("執行回測...")
    metrics = run_backtest(
        df,
        entry_signals=s.ENTRY_SIGNALS,
        exit_signals=s.EXIT_SIGNALS,
        commission=s.COMMISSION,
    )

    print_summary(metrics, params, s.ENTRY_SIGNALS, s.EXIT_SIGNALS)
    return metrics


if __name__ == "__main__":
    main()
