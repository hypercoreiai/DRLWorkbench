# Tables for metrics and reports (V3)

from typing import Any, Dict, List, Optional

import pandas as pd


def format_metrics_table(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    decimals: int = 4,
) -> str:
    """
    Format a metrics DataFrame for display (e.g. markdown or plain text).

    Args:
        df: Metrics DataFrame.
        columns: Columns to include (default all).
        decimals: Decimal places for numeric columns.

    Returns:
        Formatted string.
    """
    if df.empty:
        return ""
    out = df.copy()
    if columns:
        out = out[[c for c in columns if c in out.columns]]
    for c in out.select_dtypes(include=["number"]).columns:
        out[c] = out[c].round(decimals)
    return out.to_string()


def summary_table_from_backtest(
    report: Dict[str, Any],
    strategies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build summary table from backtest report."""
    rows = []
    for k, v in report.items():
        if isinstance(v, dict) and "sharpe" in v:
            row = {"strategy": k, **v}
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()
