"""Forecasting metrics for multi-step predictions."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "smape": np.nan,
            "r2": np.nan,
        }
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "smape": smape, "r2": r2}


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return np.nan
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def coverage_rate(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if y_true.size == 0:
        return np.nan
    return np.mean((y_true >= lower) & (y_true <= upper))


def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute MAE and RMSE per horizon."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 2:
        raise ValueError("y_true must be 2D: (n_samples, horizon)")
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return {"mae": mae, "rmse": rmse}


def information_ratio(
    predicted_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    annualization: float = 252.0,
) -> float:
    predicted_returns = np.asarray(predicted_returns)
    benchmark_returns = np.asarray(benchmark_returns)
    if predicted_returns.size == 0 or benchmark_returns.size == 0:
        return np.nan
    excess = predicted_returns - benchmark_returns
    tracking_error = np.std(excess)
    if tracking_error == 0:
        return np.nan
    return np.mean(excess) / tracking_error * np.sqrt(annualization)


def calibration_curve(nominals: Tuple[float, ...], coverages: Tuple[float, ...]) -> Dict[str, np.ndarray]:
    return {"nominal": np.array(nominals), "empirical": np.array(coverages)}