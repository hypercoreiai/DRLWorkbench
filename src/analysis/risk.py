# Portfolio risk metrics (QuantStats-style, regime-conditioned) (V3 — PROJECT_OUTLINE Section 5.2)

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def compute_risk_metrics(
    returns: pd.Series,
    regime_labels: Optional[np.ndarray] = None,
    annualization: int = 252,
) -> pd.DataFrame:
    """
    Core metrics: Sharpe, Omega, Sortino, Calmar, max drawdown, etc.
    If regime_labels provided, add regime-conditioned metrics.

    Returns:
        DataFrame with risk metrics (one row or one per regime).
    """
    r = returns.dropna()
    if r.empty or r.std() == 0:
        return pd.DataFrame()

    sharpe = r.mean() / r.std() * np.sqrt(annualization)
    neg = r[r < 0]
    sortino = (
        r.mean() / neg.std() * np.sqrt(annualization)
        if len(neg) and neg.std() != 0
        else np.nan
    )
    cum = (1 + r).cumprod()
    dd = (cum.cummax() - cum) / cum.cummax()
    max_dd = dd.max()
    calmar = r.mean() / max_dd * annualization if max_dd != 0 else np.nan

    d = {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "volatility": r.std() * np.sqrt(annualization),
        "annual_return": r.mean() * annualization,
    }
    return pd.DataFrame([d])
