from __future__ import annotations

"""
Autoresearch trading baseline for GOOGL daily bars.

This file is intentionally the mutable search surface. The agent can change model
architecture, optimization, regularization, lookback, and output mapping, while
prepare.py and the CSV stay fixed.
"""

import os
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from prepare import (
    ALLOW_SHORT,
    TIME_BUDGET,
    TRANSACTION_COST_BPS,
    backtest_positions,
    build_locked_test_split,
    build_search_folds,
    describe_dataset,
    summarize_metrics,
)

# -----------------------------------------------------------------------------
# Mutable search-space defaults
# -----------------------------------------------------------------------------

LOOKBACK = 20
HIDDEN_DIM = 64
DEPTH = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 35
TURNOVER_REG = 0.03
LOGIT_L2_REG = 1e-4
SEED = 42


@dataclass
class FoldResult:
    name: str
    best_val_sharpe: float
    best_epoch: int
    turnover: float
    max_drawdown: float
    exposure: float
    cagr: float
    trades: int


class PolicyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, depth: int = DEPTH):
        super().__init__()
        self.hidden_layers = []
        in_dim = input_dim
        for _ in range(depth):
            layer = nn.Linear(in_dim, hidden_dim)
            self.hidden_layers.append(layer)
            in_dim = hidden_dim
        self.out = nn.Linear(in_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = x.reshape((x.shape[0], -1))
        for layer in self.hidden_layers:
            h = nn.gelu(layer(h))
        logits = self.out(h)
        return logits.squeeze(-1)


def positions_from_logits(logits: mx.array) -> mx.array:
    if ALLOW_SHORT:
        return mx.tanh(logits)
    return mx.sigmoid(logits)


def _cost_rate() -> float:
    return TRANSACTION_COST_BPS * 1e-4


def train_loss(model: PolicyMLP, x: mx.array, y: mx.array) -> tuple[mx.array, dict[str, mx.array]]:
    logits = model(x)
    pos = positions_from_logits(logits)
    prev_pos = mx.concatenate([mx.zeros((1,), dtype=pos.dtype), pos[:-1]], axis=0)
    turnover = mx.abs(pos - prev_pos)
    net = pos * y - _cost_rate() * turnover
    mean_ret = mx.mean(net)
    centered = net - mean_ret
    vol = mx.sqrt(mx.mean(centered * centered) + 1e-8)
    sharpe = mx.sqrt(mx.array(252.0, dtype=net.dtype)) * mean_ret / vol
    loss = -sharpe + TURNOVER_REG * mx.mean(turnover) + LOGIT_L2_REG * mx.mean(logits * logits)
    metrics = {
        "train_sharpe": sharpe,
        "train_turnover": mx.mean(turnover),
        "train_exposure": mx.mean(mx.abs(pos)),
    }
    return loss, metrics


def _to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x)


def evaluate_fold(model: PolicyMLP, x_val: np.ndarray, y_val: np.ndarray) -> tuple[float, dict[str, float]]:
    logits = model(mx.array(x_val))
    pos = positions_from_logits(logits)
    pos_np = _to_numpy(pos)
    metrics, _ = backtest_positions(pos_np, y_val, cost_bps=TRANSACTION_COST_BPS, allow_short=ALLOW_SHORT)
    return metrics.sharpe, {
        "turnover": metrics.turnover,
        "max_drawdown": metrics.max_drawdown,
        "exposure": metrics.exposure,
        "cagr": metrics.cagr,
        "trades": float(metrics.trades),
    }


def _new_model(input_dim: int) -> PolicyMLP:
    return PolicyMLP(input_dim=input_dim, hidden_dim=HIDDEN_DIM, depth=DEPTH)


def _new_optimizer() -> optim.AdamW:
    return optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def train_one_fold(fold, budget_seconds: float) -> FoldResult:
    input_dim = int(fold.X_train.shape[1] * fold.X_train.shape[2])
    model = _new_model(input_dim=input_dim)
    optimizer = _new_optimizer()
    loss_and_grad = nn.value_and_grad(model, train_loss)

    x_train = mx.array(fold.X_train)
    y_train = mx.array(fold.y_train)

    best_val_sharpe = -1e9
    best_epoch = 0
    best_metrics = {
        "turnover": 0.0,
        "max_drawdown": 0.0,
        "exposure": 0.0,
        "cagr": 0.0,
        "trades": 0.0,
    }
    patience_left = PATIENCE
    started = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        (loss, train_metrics), grads = loss_and_grad(model, x_train, y_train)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        val_sharpe, val_metrics = evaluate_fold(model, fold.X_val, fold.y_val)
        improved = val_sharpe > best_val_sharpe + 1e-6
        if improved:
            best_val_sharpe = val_sharpe
            best_epoch = epoch
            best_metrics = val_metrics
            patience_left = PATIENCE
        else:
            patience_left -= 1

        elapsed = time.time() - started
        if epoch == 1 or epoch % 25 == 0 or improved:
            train_sharpe = float(_to_numpy(train_metrics["train_sharpe"]))
            print(
                f"{fold.name} epoch={epoch:03d} "
                f"train_sharpe={train_sharpe: .4f} "
                f"val_sharpe={val_sharpe: .4f} "
                f"elapsed={elapsed: .1f}s"
            )

        if patience_left <= 0:
            break
        if elapsed >= budget_seconds:
            break

    return FoldResult(
        name=fold.name,
        best_val_sharpe=float(best_val_sharpe),
        best_epoch=int(best_epoch),
        turnover=float(best_metrics["turnover"]),
        max_drawdown=float(best_metrics["max_drawdown"]),
        exposure=float(best_metrics["exposure"]),
        cagr=float(best_metrics["cagr"]),
        trades=int(best_metrics["trades"]),
    )


def train_final_model(x_train_np: np.ndarray, y_train_np: np.ndarray, budget_seconds: float) -> PolicyMLP:
    """
    Fit one final model on the full search-train split without looking at the
    locked test during optimization.
    """
    input_dim = int(x_train_np.shape[1] * x_train_np.shape[2])
    model = _new_model(input_dim=input_dim)
    optimizer = _new_optimizer()
    loss_and_grad = nn.value_and_grad(model, train_loss)

    x_train = mx.array(x_train_np)
    y_train = mx.array(y_train_np)
    started = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        (loss, train_metrics), grads = loss_and_grad(model, x_train, y_train)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        elapsed = time.time() - started

        if epoch == 1 or epoch % 25 == 0:
            train_sharpe = float(_to_numpy(train_metrics["train_sharpe"]))
            print(
                f"locked_test_train epoch={epoch:03d} "
                f"train_sharpe={train_sharpe: .4f} "
                f"elapsed={elapsed: .1f}s"
            )

        if elapsed >= budget_seconds:
            break

    return model


def maybe_run_locked_test() -> None:
    if os.environ.get("REVEAL_FINAL_TEST") != "1":
        return

    print()
    print("Running locked final test...")
    locked = build_locked_test_split(LOOKBACK)
    model = train_final_model(locked.X_train, locked.y_train, budget_seconds=max(10.0, float(TIME_BUDGET)))
    locked_sharpe, locked_metrics = evaluate_fold(model, locked.X_val, locked.y_val)

    print()
    print("--- LOCKED FINAL TEST ---")
    print(f"locked_test_name:        {locked.name}")
    print(f"locked_test_start:       {locked.val_dates[0]}")
    print(f"locked_test_end:         {locked.val_dates[-1]}")
    print(f"locked_test_sharpe:      {locked_sharpe:.6f}")
    print(f"locked_test_turnover:    {locked_metrics['turnover']:.6f}")
    print(f"locked_test_drawdown:    {locked_metrics['max_drawdown']:.6f}")
    print(f"locked_test_cagr:        {locked_metrics['cagr']:.6f}")
    print(f"locked_test_exposure:    {locked_metrics['exposure']:.6f}")
    print(f"locked_test_trades:      {locked_metrics['trades']:.1f}")


def main() -> None:
    np.random.seed(SEED)
    try:
        mx.random.seed(SEED)
    except Exception:
        pass

    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass

    started = time.time()
    dataset_info = describe_dataset(LOOKBACK)
    folds = build_search_folds(LOOKBACK)

    print("Dataset summary:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    print()

    per_fold_budget = max(10.0, TIME_BUDGET / max(len(folds), 1))
    fold_results: list[FoldResult] = []

    for fold in folds:
        result = train_one_fold(fold, budget_seconds=per_fold_budget)
        fold_results.append(result)

    pseudo_metrics = []
    for result in fold_results:
        pseudo_metrics.append(
            type(
                "PseudoMetric",
                (),
                {
                    "sharpe": result.best_val_sharpe,
                    "cagr": result.cagr,
                    "max_drawdown": result.max_drawdown,
                    "turnover": result.turnover,
                    "exposure": result.exposure,
                    "trades": result.trades,
                },
            )()
        )
    summary = summarize_metrics(pseudo_metrics)

    total_seconds = time.time() - started
    peak_vram_mb = 0.0
    try:
        peak_vram_mb = float(mx.metal.get_peak_memory()) / (1024.0 * 1024.0)
    except Exception:
        peak_vram_mb = 0.0

    num_params = 0
    try:
        baseline_model = _new_model(
            input_dim=int(folds[0].X_train.shape[1] * folds[0].X_train.shape[2]),
        )
        params = baseline_model.parameters()
        flat_params = []

        def _flatten(tree):
            if isinstance(tree, dict):
                for value in tree.values():
                    _flatten(value)
            elif isinstance(tree, (list, tuple)):
                for value in tree:
                    _flatten(value)
            else:
                flat_params.append(tree)

        _flatten(params)
        num_params = int(sum(int(np.prod(p.shape)) for p in flat_params))
    except Exception:
        num_params = 0

    print()
    print("---")
    print(f"val_sharpe:        {summary['val_sharpe']:.6f}")
    print(f"training_seconds:  {total_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"median_turnover:   {summary['median_turnover']:.6f}")
    print(f"median_drawdown:   {summary['median_max_drawdown']:.6f}")
    print(f"median_cagr:       {summary['median_cagr']:.6f}")
    print(f"median_exposure:   {summary['median_exposure']:.6f}")
    print(f"median_trades:     {summary['median_trades']:.1f}")
    print(f"num_params:        {num_params}")
    print(f"lookback:          {LOOKBACK}")
    print(f"folds:             {len(fold_results)}")
    print(
        "fold_sharpes:      "
        + ", ".join(f"{r.name}={r.best_val_sharpe:.4f}" for r in fold_results)
    )

    maybe_run_locked_test()


if __name__ == "__main__":
    main()
