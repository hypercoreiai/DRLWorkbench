# Plots: equity curve, drawdown, rolling metrics, regime, diagnostics (V3 — PROJECT_OUTLINE Section 6)

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def plot_equity_curve_with_regimes(
    returns: pd.Series,
    regime_labels: Optional[np.ndarray] = None,
    strategies: Optional[Dict[str, pd.Series]] = None,
    ax: Any = None,
) -> Any:
    """Equity curve over time, with regime backgrounds (colors)."""
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    cum = (1 + returns).cumprod()
    ax.plot(cum.index, cum.values, label="portfolio")
    if strategies:
        for name, s in strategies.items():
            c = (1 + s).cumprod()
            ax.plot(c.index, c.values, label=name)
    ax.legend()
    ax.set_title("Equity curve")
    return ax


def plot_drawdown_analysis(
    returns: pd.Series,
    strategies: Optional[Dict[str, pd.Series]] = None,
    ax: Any = None,
) -> Any:
    """Drawdown chart: running max / current value, highlight max DD."""
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    cum = (1 + returns).cumprod()
    dd = (cum.cummax() - cum) / cum.cummax()
    ax.fill_between(dd.index, 0, dd.values, alpha=0.3)
    ax.set_title("Drawdown")
    return ax


def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 60,
    metrics: Optional[List[str]] = None,
    ax: Any = None,
) -> Any:
    """Rolling Sharpe, volatility, correlation over time."""
    if metrics is None:
        metrics = ["sharpe", "volatility"]
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    roll = returns.rolling(window, min_periods=1)
    if "sharpe" in metrics:
        sh = roll.mean() / roll.std() * np.sqrt(252)
        ax.plot(sh.index, sh.values, label="rolling_sharpe")
    if "volatility" in metrics:
        vol = roll.std() * np.sqrt(252)
        ax.plot(vol.index, vol.values, label="rolling_volatility")
    ax.legend()
    ax.set_title("Rolling metrics")
    return ax


def plot_strategy_comparison_dashboard(
    results_dict: Dict[str, Any],
    ax: Any = None,
) -> Any:
    """Summary table / multi-strategy overlay placeholder."""
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    ax.text(0.5, 0.5, "Strategy comparison dashboard", ha="center", va="center")
    return ax


def plot_residuals_diagnostic(residuals: np.ndarray, ax: Any = None) -> Any:
    """Residuals: time series, histogram, Q-Q (placeholder)."""
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    ax.hist(residuals, bins=min(50, len(residuals) // 5), density=True)
    ax.set_title("Residuals histogram")
    return ax
