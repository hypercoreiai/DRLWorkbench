# Regime detection and regime-aware metrics (V3 — PROJECT_OUTLINE Section 4.3)

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class RegimeDetector:
    """
    Market regime detection using various methods.
    Supports volatility-based, return-based, and clustering methods.
    """
    
    def __init__(self, method: str = "volatility", n_regimes: int = 3, **kwargs):
        """
        Args:
            method: Detection method ('volatility', 'return', 'kmeans', 'gmm', 'rule_based')
            n_regimes: Number of regimes to detect
            **kwargs: Additional method-specific parameters
        """
        self.method = method
        self.n_regimes = n_regimes
        self.kwargs = kwargs
        self.fitted_ = False
        
    def fit_predict(self, returns: pd.Series) -> np.ndarray:
        """
        Fit regime detector and predict regime labels.
        
        Args:
            returns: Time series of returns
        
        Returns:
            Array of regime labels (0, 1, 2, ...)
        """
        if self.method in ['volatility', 'return']:
            window = self.kwargs.get('window', 63)
            regimes = detect_regimes(returns, method=self.method, window=window)
        elif self.method == 'kmeans':
            regimes = self._kmeans_regimes(returns)
        elif self.method == 'gmm':
            regimes = self._gmm_regimes(returns)
        elif self.method == 'rule_based':
            regimes = self._rule_based_regimes(returns)
        else:
            # Default to volatility
            regimes = detect_regimes(returns, method='volatility')
        
        self.fitted_ = True
        return regimes
    
    def _kmeans_regimes(self, returns: pd.Series) -> np.ndarray:
        """K-Means clustering on returns and volatility."""
        try:
            from sklearn.cluster import KMeans
            
            # Create features: returns and rolling volatility
            vol = returns.rolling(20, min_periods=1).std()
            features = np.column_stack([returns.values, vol.values])
            
            # Remove NaN
            valid_idx = ~np.isnan(features).any(axis=1)
            features_valid = features[valid_idx]
            
            # Cluster
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
            labels = np.full(len(returns), 0)
            labels[valid_idx] = kmeans.fit_predict(features_valid)
            
            return labels
        except ImportError:
            # Fallback to volatility method
            return detect_regimes(returns, method='volatility')
    
    def _gmm_regimes(self, returns: pd.Series) -> np.ndarray:
        """Gaussian Mixture Model clustering."""
        try:
            from sklearn.mixture import GaussianMixture
            
            # Create features
            vol = returns.rolling(20, min_periods=1).std()
            features = np.column_stack([returns.values, vol.values])
            
            # Remove NaN
            valid_idx = ~np.isnan(features).any(axis=1)
            features_valid = features[valid_idx]
            
            # Cluster
            gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
            labels = np.full(len(returns), 0)
            labels[valid_idx] = gmm.fit_predict(features_valid)
            
            return labels
        except ImportError:
            return detect_regimes(returns, method='volatility')
    
    def _rule_based_regimes(self, returns: pd.Series) -> np.ndarray:
        """Rule-based regime classification (Bull/Bear/Sideways)."""
        window = self.kwargs.get('window', 63)
        
        # Rolling mean and volatility
        rolling_mean = returns.rolling(window, min_periods=1).mean()
        rolling_vol = returns.rolling(window, min_periods=1).std()
        
        # Thresholds
        mean_threshold = rolling_mean.quantile(0.3)
        vol_threshold = rolling_vol.quantile(0.5)
        
        # Classify
        labels = np.zeros(len(returns), dtype=int)
        
        # Bear: negative returns
        labels[rolling_mean < mean_threshold] = 0
        
        # Bull: positive returns, low vol
        labels[(rolling_mean >= mean_threshold) & (rolling_vol < vol_threshold)] = 1
        
        # Sideways/Volatile: positive returns, high vol
        labels[(rolling_mean >= mean_threshold) & (rolling_vol >= vol_threshold)] = 2
        
        return labels


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
