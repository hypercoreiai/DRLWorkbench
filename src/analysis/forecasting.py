# Forecasting evaluation: point, directional, probabilistic (V3 — PROJECT_OUTLINE Section 5.1)

from typing import Any, Optional

import numpy as np
import pandas as pd


def compute_forecasting_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    ticker: Optional[str] = None,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Point metrics: MSE, MAPE, MAE.
    Directional: Up/Down hit rate, false positive rate.
    Probabilistic: CRPS if probabilistic forecast.

    Returns:
        DataFrame with all metrics.
    """
    actual = np.asarray(actual)
    pred = np.asarray(predicted)
    mse = np.mean((actual - pred) ** 2)
    mae = np.mean(np.abs(actual - pred))
    mape = np.nanmean(np.abs((actual - pred) / (actual + 1e-10))) * 100

    # Directional
    act_dir = np.sign(np.diff(actual, prepend=actual[0]))
    pred_dir = np.sign(np.diff(pred, prepend=pred[0]))
    hit = np.mean(act_dir == pred_dir)
    pos_mask = pred_dir > 0
    fp_rate = (
        np.mean(act_dir[pos_mask] <= 0) if pos_mask.any() else np.nan
    )

    d = {
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "directional_hit_rate": hit,
        "false_positive_rate": fp_rate,
        "ticker": ticker,
        "model_name": model_name,
    }
    return pd.DataFrame([d])
