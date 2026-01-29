# Regime detection and regime-aware metrics (V3 — PROJECT_OUTLINE Section 4.3)

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def detect_regimes(
    returns: pd.Series,
    method: str = "volatility",
    window: int = 63,
    percentiles: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Detect market regimes (e.g. low/medium/high vol, bull/bear/sideways).

    Args:
        returns: Time series of returns.
        method: 'volatility' or 'return'.
        window: Rolling window size.
        percentiles: Bounds for regime bins (e.g. [33, 66]).

    Returns:
        Array of regime labels (0, 1, 2, ...).
    """
    if percentiles is None:
        percentiles = [33.33, 66.67]
    if method == "volatility":
        vol = returns.rolling(window, min_periods=1).std()
        bounds = np.nanpercentile(vol.dropna(), percentiles)
        labels = np.searchsorted(bounds, vol.values, side="right")
        return np.clip(labels, 0, len(percentiles))
    if method == "return":
        mu = returns.rolling(window, min_periods=1).mean()
        bounds = np.nanpercentile(mu.dropna(), percentiles)
        labels = np.searchsorted(bounds, mu.values, side="right")
        return np.clip(labels, 0, len(percentiles))
    return np.zeros(len(returns), dtype=int)


def compute_regime_metrics(
    returns: pd.Series, regime_labels: np.ndarray
) -> pd.DataFrame:
    """
    For each regime: Sharpe, Sortino, max drawdown, win rate, volatility.

    Returns:
        DataFrame with regime breakdown.
    """
    results = []
    returns_arr = np.asarray(returns)
    for r in np.unique(regime_labels):
        mask = regime_labels == r
        r_ = pd.Series(returns_arr[mask])
        if r_.empty or r_.std() == 0:
            results.append(
                {
                    "regime": r,
                    "sharpe": np.nan,
                    "sortino": np.nan,
                    "max_drawdown": np.nan,
                    "win_rate": np.nan,
                    "volatility": np.nan,
                }
            )
            continue
        sharpe = r_.mean() / r_.std() * np.sqrt(252) if r_.std() != 0 else np.nan
        neg = r_[r_ < 0]
        sortino = (
            r_.mean() / neg.std() * np.sqrt(252)
            if len(neg) and neg.std() != 0
            else np.nan
        )
        cum = (1 + r_).cumprod()
        dd = (cum.cummax() - cum) / cum.cummax()
        results.append(
            {
                "regime": r,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": dd.max(),
                "win_rate": (r_ > 0).mean(),
                "volatility": r_.std() * np.sqrt(252),
            }
        )
    return pd.DataFrame(results)
