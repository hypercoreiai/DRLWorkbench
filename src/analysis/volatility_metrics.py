"""
Volatility Forecasting Metrics and Analysis
Specialized metrics for evaluating volatility forecasts
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


def compute_volatility_forecast_metrics(
    actual_vol: np.ndarray,
    predicted_vol: np.ndarray,
    returns: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive volatility forecast metrics.
    
    Args:
        actual_vol: Actual realized volatility
        predicted_vol: Predicted volatility
        returns: Optional returns for additional metrics
    
    Returns:
        Dictionary of metrics
    """
    actual = np.asarray(actual_vol).flatten()
    pred = np.asarray(predicted_vol).flatten()
    
    # Ensure same length
    min_len = min(len(actual), len(pred))
    actual = actual[:min_len]
    pred = pred[:min_len]
    
    # Remove NaN
    valid_mask = ~(np.isnan(actual) | np.isnan(pred))
    actual = actual[valid_mask]
    pred = pred[valid_mask]
    
    if len(actual) < 2:
        return {metric: np.nan for metric in [
            'mse', 'rmse', 'mae', 'mape', 'r2', 'qlike', 'log_loss'
        ]}
    
    # Mean Squared Error
    mse = np.mean((actual - pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - pred))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-10))) * 100
    
    # R-squared
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # QLIKE (Quasi-Likelihood) - asymmetric loss function
    # Penalizes underestimation more than overestimation
    qlike = np.mean((actual / (pred + 1e-10)) - np.log(actual / (pred + 1e-10)) - 1)
    
    # Log loss (for comparing log volatilities)
    log_loss = np.mean((np.log(actual + 1e-10) - np.log(pred + 1e-10)) ** 2)
    
    # Direction accuracy (for changes in volatility)
    if len(actual) > 1:
        actual_direction = np.sign(np.diff(actual))
        pred_direction = np.sign(np.diff(pred))
        direction_accuracy = np.mean(actual_direction == pred_direction)
    else:
        direction_accuracy = np.nan
    
    # Bias
    bias = np.mean(pred - actual)
    
    # Correlation
    correlation = np.corrcoef(actual, pred)[0, 1] if len(actual) > 1 else np.nan
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'qlike': float(qlike),
        'log_loss': float(log_loss),
        'direction_accuracy': float(direction_accuracy) if not np.isnan(direction_accuracy) else np.nan,
        'bias': float(bias),
        'correlation': float(correlation) if not np.isnan(correlation) else np.nan
    }
    
    # VaR coverage if returns provided
    if returns is not None:
        var_metrics = compute_var_coverage(returns[:len(pred)], pred)
        metrics.update(var_metrics)
    
    return metrics


def compute_var_coverage(
    returns: np.ndarray,
    predicted_vol: np.ndarray,
    confidence_levels: list = None
) -> Dict[str, float]:
    """
    Compute VaR coverage metrics.
    Check if actual returns fall within predicted confidence intervals.
    
    Args:
        returns: Actual returns
        predicted_vol: Predicted volatility (annualized)
        confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
    
    Returns:
        Dictionary with coverage ratios for each confidence level
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]
    
    returns = np.asarray(returns).flatten()
    pred_vol = np.asarray(predicted_vol).flatten()
    
    # De-annualize volatility to daily
    pred_vol_daily = pred_vol / np.sqrt(252)
    
    metrics = {}
    
    for conf in confidence_levels:
        # Calculate VaR threshold (normal assumption)
        from scipy.stats import norm
        z_score = norm.ppf(1 - (1 - conf) / 2)  # Two-tailed
        
        var_threshold = z_score * pred_vol_daily
        
        # Check coverage
        violations = np.abs(returns) > var_threshold
        actual_coverage = 1 - np.mean(violations)
        
        # Coverage ratio (actual/expected)
        coverage_ratio = actual_coverage / conf if conf > 0 else np.nan
        
        metrics[f'var_coverage_{int(conf*100)}'] = float(actual_coverage)
        metrics[f'var_coverage_ratio_{int(conf*100)}'] = float(coverage_ratio)
        metrics[f'var_violations_{int(conf*100)}'] = int(np.sum(violations))
    
    return metrics


def compute_model_confidence_intervals(
    predictions: np.ndarray,
    actual: np.ndarray,
    confidence: float = 0.95
) -> Dict[str, np.ndarray]:
    """
    Compute confidence intervals for volatility forecasts.
    
    Args:
        predictions: Predicted volatility
        actual: Actual volatility
        confidence: Confidence level
    
    Returns:
        Dictionary with lower and upper bounds
    """
    residuals = actual - predictions
    std_residual = np.std(residuals)
    
    from scipy.stats import norm
    z_score = norm.ppf((1 + confidence) / 2)
    
    lower_bound = predictions - z_score * std_residual
    upper_bound = predictions + z_score * std_residual
    
    # Calculate coverage
    in_bounds = (actual >= lower_bound) & (actual <= upper_bound)
    actual_coverage = np.mean(in_bounds)
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'actual_coverage': float(actual_coverage),
        'expected_coverage': confidence
    }


def compute_diebold_mariano_test(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    loss_function: str = 'mse'
) -> Dict[str, float]:
    """
    Diebold-Mariano test for comparing two forecasting models.
    
    Args:
        actual: Actual values
        forecast1: Forecasts from model 1
        forecast2: Forecasts from model 2
        loss_function: 'mse' or 'mae'
    
    Returns:
        Dictionary with test statistic and p-value
    """
    # Calculate losses
    if loss_function == 'mse':
        loss1 = (actual - forecast1) ** 2
        loss2 = (actual - forecast2) ** 2
    elif loss_function == 'mae':
        loss1 = np.abs(actual - forecast1)
        loss2 = np.abs(actual - forecast2)
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
    
    # Loss differential
    d = loss1 - loss2
    
    # Mean and variance of differential
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    
    if d_var == 0:
        return {'dm_statistic': 0.0, 'p_value': 1.0, 'model1_better': False}
    
    # DM test statistic
    n = len(d)
    dm_stat = d_mean / np.sqrt(d_var / n)
    
    # P-value (two-tailed test)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    
    return {
        'dm_statistic': float(dm_stat),
        'p_value': float(p_value),
        'model1_better': bool(dm_stat < 0),  # Negative means model 1 has lower loss
        'significant': bool(p_value < 0.05)
    }


def compute_volatility_persistence(realized_vol: pd.Series) -> Dict[str, float]:
    """
    Measure volatility persistence (autocorrelation).
    
    Args:
        realized_vol: Time series of realized volatility
    
    Returns:
        Dictionary with persistence metrics
    """
    vol = realized_vol.dropna()
    
    if len(vol) < 3:
        return {
            'acf_lag1': np.nan,
            'acf_lag5': np.nan,
            'acf_lag22': np.nan,
            'half_life': np.nan
        }
    
    # Autocorrelation at different lags
    acf_lag1 = vol.autocorr(lag=1)
    acf_lag5 = vol.autocorr(lag=5)
    acf_lag22 = vol.autocorr(lag=22)
    
    # Half-life of mean reversion (from AR(1) coefficient)
    if acf_lag1 > 0 and acf_lag1 < 1:
        half_life = -np.log(2) / np.log(acf_lag1)
    else:
        half_life = np.nan
    
    return {
        'acf_lag1': float(acf_lag1) if not np.isnan(acf_lag1) else np.nan,
        'acf_lag5': float(acf_lag5) if not np.isnan(acf_lag5) else np.nan,
        'acf_lag22': float(acf_lag22) if not np.isnan(acf_lag22) else np.nan,
        'half_life': float(half_life) if not np.isnan(half_life) else np.nan
    }


def compute_volatility_asymmetry(
    returns: pd.Series,
    realized_vol: pd.Series
) -> Dict[str, float]:
    """
    Measure leverage effect / volatility asymmetry.
    Negative returns tend to increase volatility more than positive returns.
    
    Args:
        returns: Time series of returns
        realized_vol: Time series of realized volatility
    
    Returns:
        Dictionary with asymmetry metrics
    """
    # Align data
    common_idx = returns.index.intersection(realized_vol.index)
    ret = returns.loc[common_idx]
    vol = realized_vol.loc[common_idx]
    
    # Lagged returns
    ret_lag = ret.shift(1)
    
    # Split by sign of return
    pos_mask = ret_lag > 0
    neg_mask = ret_lag < 0
    
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        # Average volatility following positive vs negative returns
        vol_after_pos = vol[pos_mask].mean()
        vol_after_neg = vol[neg_mask].mean()
        
        # Asymmetry ratio
        asymmetry_ratio = vol_after_neg / vol_after_pos if vol_after_pos > 0 else np.nan
        
        # Correlation between lagged returns and volatility
        corr_neg_ret_vol = ret_lag[ret_lag < 0].corr(vol[ret_lag < 0])
        corr_pos_ret_vol = ret_lag[ret_lag > 0].corr(vol[ret_lag > 0])
    else:
        vol_after_pos = np.nan
        vol_after_neg = np.nan
        asymmetry_ratio = np.nan
        corr_neg_ret_vol = np.nan
        corr_pos_ret_vol = np.nan
    
    return {
        'vol_after_positive_ret': float(vol_after_pos) if not np.isnan(vol_after_pos) else np.nan,
        'vol_after_negative_ret': float(vol_after_neg) if not np.isnan(vol_after_neg) else np.nan,
        'asymmetry_ratio': float(asymmetry_ratio) if not np.isnan(asymmetry_ratio) else np.nan,
        'corr_neg_ret_vol': float(corr_neg_ret_vol) if not np.isnan(corr_neg_ret_vol) else np.nan,
        'corr_pos_ret_vol': float(corr_pos_ret_vol) if not np.isnan(corr_pos_ret_vol) else np.nan
    }
