# Comparative strategy analysis (V3 — PROJECT_OUTLINE Section 5.3)

from typing import Any, Dict

import pandas as pd


def compare_strategies(results_dict: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Input: {strategy_name: strategy_results, ...}

    Outputs:
    1. Summary table: Strategy name, annual return, volatility, Sharpe, max DD, etc.
    2. Rolling metrics (placeholder).
    3. Drawdown comparison (placeholder).
    4. Regime performance (placeholder).
    5. Correlation of strategy returns (placeholder).
    6. Sensitivity (placeholder).

    Returns:
        Structured dict of DataFrames for display.
    """
    summary_rows = []
    for name, res in results_dict.items():
        row = {"strategy": name}
        for k, v in res.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
        summary_rows.append(row)

    return {
        "summary": pd.DataFrame(summary_rows),
        "rolling_metrics": pd.DataFrame(),
        "drawdown_comparison": pd.DataFrame(),
        "regime_performance": pd.DataFrame(),
        "correlation": pd.DataFrame(),
        "sensitivity": pd.DataFrame(),
    }
