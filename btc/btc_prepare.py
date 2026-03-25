"""
btc_prepare.py — 固定不修改
一次性下載並快取 BTC/USDT 5m 歷史數據。

使用方式:
    pip install ccxt pandas numpy
    python btc_prepare.py

數據存於 btc/cache/btc_5m.csv
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime, timezone, timedelta

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "btc_5m.csv")
MONTHS = 6
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"


def download_btc_5m(months: int = MONTHS) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError:
        print("請先安裝 ccxt: pip install ccxt pandas numpy")
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)
    exchange = ccxt.binance({"enableRateLimit": True})

    # 起始時間：N 個月前
    since_dt = datetime.now(timezone.utc) - timedelta(days=months * 30)
    since_ms = int(since_dt.timestamp() * 1000)

    all_ohlcv = []
    limit = 1000
    print(f"下載 {SYMBOL} {TIMEFRAME} 數據（從 {since_dt.strftime('%Y-%m-%d')} 開始）...")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since_ms, limit=limit)
        except Exception as e:
            print(f"下載錯誤: {e}")
            time.sleep(2)
            continue

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        since_ms = ohlcv[-1][0] + 5 * 60 * 1000

        pct = min(100, len(all_ohlcv) / (months * 30 * 288) * 100)
        print(f"  已下載 {len(all_ohlcv):,} 根 K 棒 ({pct:.0f}%)...", end="\r")

        if len(ohlcv) < limit:
            break

        time.sleep(exchange.rateLimit / 1000)

    print()
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop("timestamp", axis=1, inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    df.to_csv(CACHE_FILE)
    print(f"✓ 已儲存 {len(df):,} 根 K 棒 → {CACHE_FILE}")
    return df


def load_btc_data() -> pd.DataFrame:
    if not os.path.exists(CACHE_FILE):
        print(f"找不到快取檔案，請先執行: python btc_prepare.py")
        sys.exit(1)

    df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    print(f"數據載入完成：{len(df):,} 根 5m K 棒  ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df


def compute_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    根據 params 計算所有指標，回傳加上信號欄位的 DataFrame。
    此函數由 btc_backtest.py 呼叫，固定不修改。
    """
    df = df.copy()
    close = df["close"]
    open_ = df["open"]

    # === EMA ===
    df["ema5"]   = close.ewm(span=5,   adjust=False).mean()
    df["ema10"]  = close.ewm(span=10,  adjust=False).mean()
    df["ema20"]  = close.ewm(span=20,  adjust=False).mean()
    df["ema60"]  = close.ewm(span=60,  adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    # === RSI ===
    rsi_len = params["RSI_LENGTH"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=rsi_len - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_len - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # === MACD ===
    mf, ms, msig = params["MACD_FAST"], params["MACD_SLOW"], params["MACD_SIGNAL"]
    ema_fast = close.ewm(span=mf, adjust=False).mean()
    ema_slow = close.ewm(span=ms, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=msig, adjust=False).mean()
    hist = macd_line - signal_line
    df["macd_hist"] = hist

    # === ZLSMA ===
    zlen = params["ZLSMA_LENGTH"]
    sma1 = close.rolling(zlen).mean()
    sma2 = sma1.rolling(zlen).mean()
    df["zlsma"] = 2 * sma1 - sma2

    # === 輔助函數 ===
    def crossover(a, b):
        b_prev = b.shift(1) if hasattr(b, "shift") else b
        return (a > b) & (a.shift(1) <= b_prev)

    def crossunder(a, b):
        b_prev = b.shift(1) if hasattr(b, "shift") else b
        return (a < b) & (a.shift(1) >= b_prev)

    oversold  = params["RSI_OVERSOLD"]
    overbought = params["RSI_OVERBOUGHT"]

    # === MACD Bull/Bear（與 Pine Script 完全對應）===
    h = df["macd_hist"]
    df["macd_bull"] = (
        ((h > 0) & (h.shift(1) < 0)) |
        ((h > h.shift(1)) & (h.shift(1) > h.shift(2)) & (h.shift(2) > h.shift(3)))
    )
    df["macd_bear"] = (
        ((h < 0) & (h.shift(1) > 0)) |
        ((h < h.shift(1)) & (h.shift(1) < h.shift(2)) & (h.shift(2) < h.shift(3)))
    )

    # === RSI 超賣/超買結束 ===
    df["rsi_exit_oversold"]   = crossover(df["rsi"], oversold)
    df["rsi_exit_overbought"] = crossunder(df["rsi"], overbought)

    # === ZLSMA 突破 ===
    df["price_breakout_zlsma"]  = crossover(close, df["zlsma"])
    df["price_breakdown_zlsma"] = crossunder(close, df["zlsma"])

    # === EMA5 穿越 EMA10 ===
    df["ema5_cross_over_10"]  = crossover(df["ema5"], df["ema10"])
    df["ema5_cross_under_10"] = crossunder(df["ema5"], df["ema10"])

    # === 連續 2 根 K 棒方向 ===
    is_bull = close > open_
    is_bear = close < open_
    df["consecutive2_day_up"]   = is_bull & is_bull.shift(1)
    df["consecutive2_day_down"] = is_bear & is_bear.shift(1)

    return df.dropna()


def run_backtest(df: pd.DataFrame, entry_signals: list, exit_signals: list,
                 commission: float = 0.001) -> dict:
    """
    執行回測，回傳績效指標 dict。
    入場：entry_signals 中所有信號同時為 True（AND 邏輯）
    出場：exit_signals 中所有信號同時為 True（AND 邏輯）
    成交價：信號 K 棒的收盤價（下一根開盤更真實，但簡化處理）
    """
    import numpy as np

    n = len(df)
    if n < 300:
        return _empty_metrics()

    # 組合進出場條件
    entry = pd.Series(True, index=df.index)
    for sig in entry_signals:
        entry = entry & df[sig]

    exit_ = pd.Series(True, index=df.index)
    for sig in exit_signals:
        exit_ = exit_ & df[sig]

    entry = entry.values
    exit_ = exit_.values
    close = df["close"].values

    trades = []
    position = None
    equity = [1.0]
    cash = 1.0

    for i in range(1, n):
        # 先檢查出場
        if position is not None and exit_[i]:
            exit_price = close[i]
            ret = (exit_price / position["entry_price"]) - 1.0 - 2 * commission
            cash *= (1.0 + ret)
            trades.append({
                "bars": i - position["bar"],
                "ret": ret,
                "win": ret > 0,
            })
            position = None

        # 再檢查入場
        if position is None and entry[i]:
            position = {
                "bar": i,
                "entry_price": close[i] * (1.0 + commission),
            }

        # 更新 equity
        if position is not None:
            unrealized = cash * (close[i] / position["entry_price"])
            equity.append(unrealized)
        else:
            equity.append(cash)

    # 強制平倉
    if position is not None:
        exit_price = close[-1]
        ret = (exit_price / position["entry_price"]) - 1.0 - 2 * commission
        cash *= (1.0 + ret)
        trades.append({"bars": n - 1 - position["bar"], "ret": ret, "win": ret > 0})
        equity[-1] = cash

    if not trades:
        return _empty_metrics()

    equity_arr = np.array(equity)
    bar_returns = np.diff(equity_arr) / equity_arr[:-1]

    # Sharpe（5m 年化，24/7 加密貨幣）
    periods_per_year = 365 * 288
    mean_r = bar_returns.mean()
    std_r  = bar_returns.std()
    sharpe = (mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 1e-10 else 0.0

    # Max Drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - peak) / peak
    max_dd = float(abs(drawdowns.min())) * 100

    # Profit Factor
    rets = np.array([t["ret"] for t in trades])
    gross_profit = rets[rets > 0].sum() if (rets > 0).any() else 0.0
    gross_loss   = abs(rets[rets < 0].sum()) if (rets < 0).any() else 1e-10
    profit_factor = gross_profit / gross_loss

    win_rate     = sum(t["win"] for t in trades) / len(trades)
    total_return = (equity_arr[-1] - 1.0) * 100
    avg_bars     = np.mean([t["bars"] for t in trades])

    return {
        "sharpe":        round(sharpe, 4),
        "profit_factor": round(profit_factor, 4),
        "win_rate":      round(win_rate, 4),
        "max_drawdown":  round(max_dd, 2),
        "total_return":  round(total_return, 2),
        "num_trades":    len(trades),
        "avg_hold_bars": round(avg_bars, 1),
        "final_equity":  round(float(equity_arr[-1]), 6),
    }


def _empty_metrics() -> dict:
    return {
        "sharpe": -99.0, "profit_factor": 0.0, "win_rate": 0.0,
        "max_drawdown": 100.0, "total_return": -100.0,
        "num_trades": 0, "avg_hold_bars": 0.0, "final_equity": 0.0,
    }


def print_summary(metrics: dict, params: dict, entry_signals: list, exit_signals: list):
    print("---")
    print(f"sharpe:           {metrics['sharpe']:.4f}")
    print(f"profit_factor:    {metrics['profit_factor']:.4f}")
    print(f"win_rate:         {metrics['win_rate']:.2%}")
    print(f"max_drawdown_pct: {metrics['max_drawdown']:.2f}")
    print(f"total_return_pct: {metrics['total_return']:.2f}")
    print(f"num_trades:       {metrics['num_trades']}")
    print(f"avg_hold_bars:    {metrics['avg_hold_bars']:.1f}")
    print(f"entry_signals:    {entry_signals}")
    print(f"exit_signals:     {exit_signals}")
    print(f"RSI_LENGTH:       {params['RSI_LENGTH']}")
    print(f"RSI_OVERSOLD:     {params['RSI_OVERSOLD']}")
    print(f"RSI_OVERBOUGHT:   {params['RSI_OVERBOUGHT']}")
    print(f"ZLSMA_LENGTH:     {params['ZLSMA_LENGTH']}")
    print(f"MACD:             {params['MACD_FAST']}/{params['MACD_SLOW']}/{params['MACD_SIGNAL']}")


if __name__ == "__main__":
    download_btc_5m()
