"""
Volatility Forecasting Plots
Specialized visualizations for volatility analysis
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


def plot_volatility_forecast(
    actual_vol: pd.Series,
    predicted_vol: Dict[str, pd.Series],
    title: str = "Volatility Forecast",
    ax: Optional[Any] = None
) -> Any:
    """
    Plot actual vs predicted volatility for multiple models.
    
    Args:
        actual_vol: Actual realized volatility
        predicted_vol: Dict of model_name -> predicted volatility Series
        title: Plot title
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual
    ax.plot(actual_vol.index, actual_vol.values, 
            label='Actual', color='black', linewidth=2, alpha=0.7)
    
    # Plot predictions
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for idx, (model_name, pred_series) in enumerate(predicted_vol.items()):
        color = colors[idx % len(colors)]
        ax.plot(pred_series.index, pred_series.values,
                label=model_name, color=color, linewidth=1.5, alpha=0.6, linestyle='--')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_volatility_forecast_errors(
    actual_vol: pd.Series,
    predicted_vol: Dict[str, pd.Series],
    ax: Optional[Any] = None
) -> Any:
    """
    Plot forecast errors (residuals) over time.
    
    Args:
        actual_vol: Actual realized volatility
        predicted_vol: Dict of model_name -> predicted volatility
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (model_name, pred_series) in enumerate(predicted_vol.items()):
        # Align indices
        common_idx = actual_vol.index.intersection(pred_series.index)
        errors = actual_vol.loc[common_idx] - pred_series.loc[common_idx]
        
        color = colors[idx % len(colors)]
        ax.plot(errors.index, errors.values, label=model_name, 
                color=color, alpha=0.6, linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Forecast Error')
    ax.set_title('Volatility Forecast Errors')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_volatility_scatter(
    actual_vol: np.ndarray,
    predicted_vol: np.ndarray,
    model_name: str = "",
    ax: Optional[Any] = None
) -> Any:
    """
    Scatter plot of actual vs predicted volatility with 45-degree line.
    
    Args:
        actual_vol: Actual volatility
        predicted_vol: Predicted volatility
        model_name: Model name for title
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Remove NaN
    valid_mask = ~(np.isnan(actual_vol) | np.isnan(predicted_vol))
    actual = actual_vol[valid_mask]
    pred = predicted_vol[valid_mask]
    
    # Scatter plot
    ax.scatter(actual, pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    # 45-degree line
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Forecast')
    
    # Calculate R²
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    ax.set_xlabel('Actual Volatility')
    ax.set_ylabel('Predicted Volatility')
    ax.set_title(f'{model_name} Volatility Forecast\nR² = {r2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Make square
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def plot_volatility_qq(
    actual_vol: np.ndarray,
    predicted_vol: np.ndarray,
    model_name: str = "",
    ax: Optional[Any] = None
) -> Any:
    """
    Q-Q plot for forecast errors.
    
    Args:
        actual_vol: Actual volatility
        predicted_vol: Predicted volatility
        model_name: Model name
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    from scipy import stats
    
    # Calculate errors
    errors = actual_vol - predicted_vol
    errors = errors[~np.isnan(errors)]
    
    # Q-Q plot
    stats.probplot(errors, dist="norm", plot=ax)
    
    ax.set_title(f'{model_name} Forecast Errors Q-Q Plot')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_realized_volatility_comparison(
    realized_vols: Dict[str, pd.Series],
    title: str = "Realized Volatility Estimators",
    ax: Optional[Any] = None
) -> Any:
    """
    Compare different realized volatility estimators.
    
    Args:
        realized_vols: Dict of estimator_name -> volatility Series
        title: Plot title
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, (est_name, vol_series) in enumerate(realized_vols.items()):
        color = colors[idx % len(colors)]
        ax.plot(vol_series.index, vol_series.values, 
                label=est_name, color=color, alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_volatility_surface(
    dates: pd.DatetimeIndex,
    tickers: List[str],
    volatilities: np.ndarray,
    title: str = "Volatility Surface",
    ax: Optional[Any] = None
) -> Any:
    """
    Heatmap of volatility across assets and time.
    
    Args:
        dates: Date index
        tickers: List of ticker names
        volatilities: 2D array (time x assets)
        title: Plot title
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    im = ax.imshow(volatilities.T, aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest', origin='lower')
    
    # Set ticks
    n_dates = len(dates)
    date_ticks = np.linspace(0, n_dates-1, min(10, n_dates), dtype=int)
    ax.set_xticks(date_ticks)
    ax.set_xticklabels([dates[i].strftime('%Y-%m-%d') for i in date_ticks], rotation=45)
    
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Asset')
    ax.set_title(title)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Volatility')
    
    return ax


def plot_volatility_cone(
    realized_vol: pd.Series,
    percentiles: List[float] = None,
    windows: List[int] = None,
    ax: Optional[Any] = None
) -> Any:
    """
    Volatility cone showing historical volatility distribution across horizons.
    
    Args:
        realized_vol: Time series of realized volatility
        percentiles: List of percentiles to plot
        windows: List of rolling windows
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]
    
    if windows is None:
        windows = [5, 10, 20, 40, 60, 120]
    
    # Calculate percentiles for each window
    cone_data = []
    for window in windows:
        rolling_vols = []
        for i in range(len(realized_vol) - window + 1):
            rolling_vols.append(realized_vol.iloc[i:i+window].mean())
        
        if rolling_vols:
            cone_data.append([np.percentile(rolling_vols, p) for p in percentiles])
        else:
            cone_data.append([np.nan] * len(percentiles))
    
    cone_data = np.array(cone_data)
    
    # Plot percentiles
    for i, p in enumerate(percentiles):
        ax.plot(windows, cone_data[:, i], marker='o', label=f'{p}th percentile')
    
    # Current volatility
    current_vol = realized_vol.iloc[-1]
    ax.axhline(y=current_vol, color='red', linestyle='--', 
               linewidth=2, label=f'Current: {current_vol:.2f}')
    
    ax.set_xlabel('Window (days)')
    ax.set_ylabel('Annualized Volatility')
    ax.set_title('Volatility Cone')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_volatility_term_structure(
    forecasts: Dict[str, np.ndarray],
    horizons: np.ndarray,
    title: str = "Volatility Term Structure",
    ax: Optional[Any] = None
) -> Any:
    """
    Plot volatility term structure (forecasts at different horizons).
    
    Args:
        forecasts: Dict of model_name -> array of forecasts at different horizons
        horizons: Array of forecast horizons (in days)
        title: Plot title
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (model_name, forecast_array) in enumerate(forecasts.items()):
        color = colors[idx % len(colors)]
        ax.plot(horizons, forecast_array, marker='o', 
                label=model_name, color=color, linewidth=2)
    
    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel('Predicted Volatility')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_model_comparison_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = None,
    ax: Optional[Any] = None
) -> Any:
    """
    Bar chart comparing models across different metrics.
    
    Args:
        metrics_dict: Dict of model_name -> metrics dict
        metrics_to_plot: List of metric names to include
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    if metrics_to_plot is None:
        # Use all common metrics
        all_metrics = set()
        for metrics in metrics_dict.values():
            all_metrics.update(metrics.keys())
        metrics_to_plot = ['mse', 'mae', 'r2', 'qlike']
        metrics_to_plot = [m for m in metrics_to_plot if m in all_metrics]
    
    # Prepare data
    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_to_plot)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    # Plot bars
    for i, model_name in enumerate(model_names):
        metrics = metrics_dict[model_name]
        values = [metrics.get(m, 0) for m in metrics_to_plot]
        
        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax
