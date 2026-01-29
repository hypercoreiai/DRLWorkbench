# Summary and reporting (V3)

from typing import Any, Dict

import pandas as pd


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
