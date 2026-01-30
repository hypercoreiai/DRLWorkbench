# Summary and reporting (V3)

from typing import Any, Dict
import numpy as np
import pandas as pd


def compute_summary_stats(returns: pd.Series, annualization: int = 252) -> Dict[str, float]:
    """
    Compute summary statistics for returns series.
    
    Args:
        returns: Time series of returns.
        annualization: Factor for annualization (252 for daily).
    
    Returns:
        Dict with summary statistics.
    """
    r = returns.dropna()
    
    if r.empty or len(r) < 2:
        return {
            'mean': np.nan,
            'std': np.nan,
            'skew': np.nan,
            'kurtosis': np.nan,
            'annual_return': np.nan,
            'annual_volatility': np.nan,
            'min': np.nan,
            'max': np.nan,
            'count': 0
        }
    
    # Basic stats
    mean = r.mean()
    std = r.std()
    
    # Higher moments
    skew = r.skew() if len(r) > 2 else np.nan
    kurtosis = r.kurtosis() if len(r) > 3 else np.nan
    
    # Annualized stats
    annual_return = mean * annualization
    annual_volatility = std * np.sqrt(annualization)
    
    return {
        'mean': float(mean),
        'std': float(std),
        'skew': float(skew),
        'kurtosis': float(kurtosis),
        'annual_return': float(annual_return),
        'annual_volatility': float(annual_volatility),
        'min': float(r.min()),
        'max': float(r.max()),
        'count': len(r)
    }


def build_summary_table(
    results: Dict[str, Any],
    metrics: list,
) -> pd.DataFrame:
    """
    Build a summary table from backtest/analysis results.

    Args:
        results: Dict of strategy_name -> result dict.
        metrics: List of metric keys to include.

    Returns:
        DataFrame with strategies as rows, metrics as columns.
    """
    rows = []
    for name, res in results.items():
        row = {"strategy": name}
        for m in metrics:
            row[m] = res.get(m, None)
        rows.append(row)
    return pd.DataFrame(rows)
