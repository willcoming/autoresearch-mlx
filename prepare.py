from __future__ import annotations

"""
Fixed data preparation and evaluation for autoresearch-mlx trading experiments.

Expected CSV columns:
    date,open,high,low,close,volume

The search loop should treat this file and the CSV as immutable. Only train.py
is intended to mutate during the autoresearch loop.
"""

import csv
import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

# -----------------------------------------------------------------------------
# Fixed experiment constants (do not modify during search)
# -----------------------------------------------------------------------------

TIME_BUDGET = 180  # seconds for the entire train.py run
DATA_CSV = os.environ.get("AUTORESEARCH_TRADING_CSV", "data/GOOGL_1d_5y.csv")
TEST_DAYS = 252
MIN_TRAIN_DAYS = 252
N_VAL_FOLDS = 4
TRANSACTION_COST_BPS = 5.0
ANNUALIZATION = 252.0
ALLOW_SHORT = False
FEATURE_CLIP = 8.0
EPS = 1e-12


@dataclass(frozen=True)
class MarketData:
    dates: np.ndarray
    features: np.ndarray
    next_returns: np.ndarray


@dataclass(frozen=True)
class FoldData:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_dates: np.ndarray
    val_dates: np.ndarray


@dataclass(frozen=True)
class BacktestMetrics:
    sharpe: float
    cagr: float
    max_drawdown: float
    turnover: float
    exposure: float
    trades: int
    mean_daily_return: float
    daily_volatility: float


def _require_columns(fieldnames: Iterable[str] | None) -> None:
    required = {"date", "open", "high", "low", "close", "volume"}
    fields = set(fieldnames or [])
    missing = required - fields
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")


def _safe_float(value: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Non-finite numeric value encountered: {value}")
    return out


def load_price_csv(path: str = DATA_CSV) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find market data CSV at {path}. "
            "Place a file like data/GOOGL_1d_5y.csv with columns "
            "date,open,high,low,close,volume."
        )

    rows: list[tuple[str, float, float, float, float, float]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        _require_columns(reader.fieldnames)
        for row in reader:
            rows.append(
                (
                    row["date"],
                    _safe_float(row["open"]),
                    _safe_float(row["high"]),
                    _safe_float(row["low"]),
                    _safe_float(row["close"]),
                    _safe_float(row["volume"]),
                )
            )

    if len(rows) < TEST_DAYS + MIN_TRAIN_DAYS + 64:
        raise ValueError(
            f"Need more data. Found only {len(rows)} rows; expected roughly 5 years of daily bars."
        )

    rows.sort(key=lambda x: x[0])
    dates = np.array([r[0] for r in rows], dtype=object)
    open_ = np.array([r[1] for r in rows], dtype=np.float64)
    high = np.array([r[2] for r in rows], dtype=np.float64)
    low = np.array([r[3] for r in rows], dtype=np.float64)
    close = np.array([r[4] for r in rows], dtype=np.float64)
    volume = np.array([r[5] for r in rows], dtype=np.float64)
    return dates, open_, high, low, close, volume


def load_market_data(path: str = DATA_CSV) -> MarketData:
    dates, open_, high, low, close, volume = load_price_csv(path)

    prev_close = close[:-1]
    curr_open = open_[1:]
    curr_high = high[1:]
    curr_low = low[1:]
    curr_close = close[1:]
    prev_volume = np.maximum(volume[:-1], 1.0)
    curr_volume = np.maximum(volume[1:], 1.0)

    daily_ret = curr_close / np.maximum(prev_close, EPS) - 1.0
    overnight_gap = curr_open / np.maximum(prev_close, EPS) - 1.0
    intraday_range = curr_high / np.maximum(curr_low, EPS) - 1.0
    intraday_body = curr_close / np.maximum(curr_open, EPS) - 1.0
    volume_change = np.log(curr_volume / prev_volume)
    close_to_high = curr_close / np.maximum(curr_high, EPS) - 1.0
    close_to_low = curr_close / np.maximum(curr_low, EPS) - 1.0

    # Feature row t uses information available at the close of day t.
    # Target row t is the return from close_t to close_{t+1}.
    features = np.stack(
        [
            overnight_gap[:-1],
            intraday_range[:-1],
            intraday_body[:-1],
            daily_ret[:-1],
            volume_change[:-1],
            close_to_high[:-1],
            close_to_low[:-1],
        ],
        axis=1,
    )
    next_returns = daily_ret[1:]
    usable_dates = dates[1:-1]

    return MarketData(dates=usable_dates, features=features, next_returns=next_returns)


def make_windows(market: MarketData, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lookback < 2:
        raise ValueError("lookback must be >= 2")

    X, y, out_dates = [], [], []
    for end in range(lookback - 1, len(market.next_returns)):
        start = end - lookback + 1
        X.append(market.features[start : end + 1])
        y.append(market.next_returns[end])
        out_dates.append(market.dates[end])

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    d_arr = np.asarray(out_dates, dtype=object)
    return X_arr, y_arr, d_arr


def _standardize(train_X: np.ndarray, other_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = train_X.reshape(-1, train_X.shape[-1])
    mu = flat.mean(axis=0, keepdims=True)
    sigma = flat.std(axis=0, keepdims=True) + 1e-6

    train_out = np.clip((train_X - mu) / sigma, -FEATURE_CLIP, FEATURE_CLIP)
    other_out = np.clip((other_X - mu) / sigma, -FEATURE_CLIP, FEATURE_CLIP)
    return train_out.astype(np.float32), other_out.astype(np.float32)


def _build_fold_ranges(n_samples: int) -> tuple[list[tuple[int, int]], int]:
    search_end = n_samples - TEST_DAYS
    if search_end <= MIN_TRAIN_DAYS + 63:
        raise ValueError(
            "Not enough samples for search folds after holding out the final test window."
        )

    raw_val = (search_end - MIN_TRAIN_DAYS) // N_VAL_FOLDS
    val_size = max(63, raw_val)

    folds: list[tuple[int, int]] = []
    train_end = MIN_TRAIN_DAYS
    while len(folds) < N_VAL_FOLDS and train_end + val_size <= search_end:
        val_start = train_end
        val_end = val_start + val_size
        folds.append((val_start, val_end))
        train_end = val_end

    if not folds:
        raise ValueError("Failed to construct validation folds.")

    return folds, search_end


def build_search_folds(lookback: int, path: str = DATA_CSV) -> list[FoldData]:
    market = load_market_data(path)
    X, y, dates = make_windows(market, lookback)
    fold_ranges, _ = _build_fold_ranges(len(X))

    folds: list[FoldData] = []
    for idx, (val_start, val_end) in enumerate(fold_ranges, start=1):
        X_train_raw = X[:val_start]
        y_train = y[:val_start]
        X_val_raw = X[val_start:val_end]
        y_val = y[val_start:val_end]

        X_train, X_val = _standardize(X_train_raw, X_val_raw)
        fold = FoldData(
            name=f"fold_{idx}",
            X_train=X_train,
            y_train=y_train.astype(np.float32),
            X_val=X_val,
            y_val=y_val.astype(np.float32),
            train_dates=dates[:val_start],
            val_dates=dates[val_start:val_end],
        )
        folds.append(fold)

    return folds


def build_locked_test_split(lookback: int, path: str = DATA_CSV) -> FoldData:
    if os.environ.get("REVEAL_FINAL_TEST") != "1":
        raise RuntimeError(
            "Final test is locked during the autoresearch loop. "
            "Set REVEAL_FINAL_TEST=1 only after search is complete."
        )

    market = load_market_data(path)
    X, y, dates = make_windows(market, lookback)
    _, search_end = _build_fold_ranges(len(X))

    X_train_raw = X[:search_end]
    y_train = y[:search_end]
    X_test_raw = X[search_end:]
    y_test = y[search_end:]
    X_train, X_test = _standardize(X_train_raw, X_test_raw)

    return FoldData(
        name="locked_test",
        X_train=X_train,
        y_train=y_train.astype(np.float32),
        X_val=X_test,
        y_val=y_test.astype(np.float32),
        train_dates=dates[:search_end],
        val_dates=dates[search_end:],
    )


def squash_positions(raw_positions: np.ndarray, allow_short: bool = ALLOW_SHORT) -> np.ndarray:
    raw = np.asarray(raw_positions, dtype=np.float64).reshape(-1)
    raw = np.clip(raw, -30.0, 30.0)
    if allow_short:
        return np.tanh(raw)
    return 1.0 / (1.0 + np.exp(-raw))


def backtest_positions(
    positions: np.ndarray,
    next_returns: np.ndarray,
    cost_bps: float = TRANSACTION_COST_BPS,
    allow_short: bool = ALLOW_SHORT,
) -> tuple[BacktestMetrics, np.ndarray]:
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    ret = np.asarray(next_returns, dtype=np.float64).reshape(-1)
    if len(pos) != len(ret):
        raise ValueError("positions and next_returns must have the same length")

    if allow_short:
        pos = np.clip(pos, -1.0, 1.0)
    else:
        pos = np.clip(pos, 0.0, 1.0)

    prev_pos = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - prev_pos)
    costs = turnover * (cost_bps * 1e-4)
    net = pos * ret - costs

    mean_daily = float(np.mean(net)) if len(net) else 0.0
    vol_daily = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    sharpe = 0.0 if vol_daily < 1e-12 else float(np.sqrt(ANNUALIZATION) * mean_daily / vol_daily)

    equity = np.cumprod(1.0 + net)
    if len(equity) == 0:
        cagr = 0.0
        max_drawdown = 0.0
    else:
        years = max(len(net) / ANNUALIZATION, 1.0 / ANNUALIZATION)
        final_equity = max(float(equity[-1]), 1e-12)
        cagr = final_equity ** (1.0 / years) - 1.0
        peak = np.maximum.accumulate(equity)
        drawdown = 1.0 - equity / np.maximum(peak, EPS)
        max_drawdown = float(np.max(drawdown))

    exposure = float(np.mean(np.abs(pos)))
    trades = int(np.sum(turnover > 1e-6))
    metrics = BacktestMetrics(
        sharpe=sharpe,
        cagr=float(cagr),
        max_drawdown=max_drawdown,
        turnover=float(np.mean(turnover)),
        exposure=exposure,
        trades=trades,
        mean_daily_return=mean_daily,
        daily_volatility=vol_daily,
    )
    return metrics, net


def median_metric(metrics: list[BacktestMetrics], attr: str) -> float:
    values = [getattr(m, attr) for m in metrics]
    return float(np.median(np.asarray(values, dtype=np.float64)))


def summarize_metrics(metrics: list[BacktestMetrics]) -> dict[str, float]:
    return {
        "val_sharpe": median_metric(metrics, "sharpe"),
        "median_cagr": median_metric(metrics, "cagr"),
        "median_max_drawdown": median_metric(metrics, "max_drawdown"),
        "median_turnover": median_metric(metrics, "turnover"),
        "median_exposure": median_metric(metrics, "exposure"),
        "median_trades": median_metric(metrics, "trades"),
    }


def describe_dataset(lookback: int, path: str = DATA_CSV) -> dict[str, float | int | str]:
    market = load_market_data(path)
    X, y, dates = make_windows(market, lookback)
    folds, search_end = _build_fold_ranges(len(X))
    return {
        "path": path,
        "rows": len(market.dates),
        "samples": len(X),
        "features": int(X.shape[-1]),
        "lookback": lookback,
        "search_samples": search_end,
        "test_samples": len(X) - search_end,
        "folds": len(folds),
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
    }
