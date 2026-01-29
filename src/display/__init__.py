# Display: plots, tables, export, style (V3)

from .plots import (
    plot_equity_curve_with_regimes,
    plot_drawdown_analysis,
    plot_rolling_metrics,
    plot_strategy_comparison_dashboard,
    plot_residuals_diagnostic,
)
from .tables import format_metrics_table, summary_table_from_backtest
from .export import export_csv_metrics, export_backtest_report
from .style import apply_style, STYLE_DEFAULTS

__all__ = [
    "plot_equity_curve_with_regimes",
    "plot_drawdown_analysis",
    "plot_rolling_metrics",
    "plot_strategy_comparison_dashboard",
    "plot_residuals_diagnostic",
    "format_metrics_table",
    "summary_table_from_backtest",
    "export_csv_metrics",
    "export_backtest_report",
    "apply_style",
    "STYLE_DEFAULTS",
]
