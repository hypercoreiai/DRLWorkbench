"""Regime detection algorithms for identifying market states."""

import pandas as pd
import numpy as np
from typing import Optional, Literal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import logging

from drlworkbench.utils.exceptions import RegimeError

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detect market regimes (bull, bear, sideways) using various algorithms.
    
    Supports multiple detection methods including HMM, K-Means, GMM, and rule-based.
    """
    
    def __init__(
        self,
        method: Literal["kmeans", "gmm", "rule_based"] = "kmeans",
        n_regimes: int = 3
    ):
        """
        Initialize regime detector.
        
        Parameters
        ----------
        method : {"kmeans", "gmm", "rule_based"}, default "kmeans"
            Detection method to use.
        n_regimes : int, default 3
            Number of regimes to detect (typically 2-3).
        """
        self.method = method
        self.n_regimes = n_regimes
        self.model = None
        self.regime_labels = {
            0: "Bear",
            1: "Sideways",
            2: "Bull"
        }
        
        logger.info(f"RegimeDetector initialized with method={method}, n_regimes={n_regimes}")
    
    def fit_predict(
        self,
        returns: pd.Series,
        volatility_window: int = 20
    ) -> pd.Series:
        """
        Fit the model and predict regimes.
        
        Parameters
        ----------
        returns : pd.Series
            Returns series (typically daily returns).
        volatility_window : int, default 20
            Window size for volatility calculation.
            
        Returns
        -------
        pd.Series
            Series of regime labels (0, 1, 2, etc.).
            
        Raises
        ------
        RegimeError
            If regime detection fails.
        """
        try:
            logger.info(f"Detecting regimes for {len(returns)} data points")
            
            # Prepare features
            features = self._prepare_features(returns, volatility_window)
            
            # Detect regimes based on method
            if self.method == "kmeans":
                regimes = self._kmeans_detect(features)
            elif self.method == "gmm":
                regimes = self._gmm_detect(features)
            elif self.method == "rule_based":
                regimes = self._rule_based_detect(returns, features)
            else:
                raise RegimeError(f"Unknown method: {self.method}")
            
            # Sort regimes by average return (0=bear, 1=sideways, 2=bull)
            regimes = self._sort_regimes(regimes, returns)
            
            logger.info(f"Regime detection complete. Distribution: {regimes.value_counts().to_dict()}")
            
            return regimes
            
        except Exception as e:
            logger.error(f"Regime detection failed: {str(e)}")
            raise RegimeError(f"Failed to detect regimes: {str(e)}") from e
    
    def _prepare_features(
        self,
        returns: pd.Series,
        volatility_window: int
    ) -> pd.DataFrame:
        """Prepare features for regime detection."""
        features = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        features['return_20'] = returns.rolling(20).mean()
        features['return_60'] = returns.rolling(60).mean()
        
        # Rolling volatility
        features['volatility'] = returns.rolling(volatility_window).std()
        
        # Trend strength (simple linear regression R-squared)
        features['trend'] = returns.rolling(40).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=True
        )
        
        # Forward fill and drop NaN
        features = features.ffill().fillna(0)
        
        return features
    
    def _kmeans_detect(self, features: pd.DataFrame) -> pd.Series:
        """Detect regimes using K-Means clustering."""
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(features)
        self.model = kmeans
        return pd.Series(regimes, index=features.index)
    
    def _gmm_detect(self, features: pd.DataFrame) -> pd.Series:
        """Detect regimes using Gaussian Mixture Model."""
        gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
        regimes = gmm.fit_predict(features)
        self.model = gmm
        return pd.Series(regimes, index=features.index)
    
    def _rule_based_detect(
        self,
        returns: pd.Series,
        features: pd.DataFrame
    ) -> pd.Series:
        """Detect regimes using rule-based logic."""
        regimes = pd.Series(1, index=returns.index)  # Default: sideways
        
        # Bull market: positive trend and low/medium volatility
        bull_mask = (features['return_20'] > 0.001) & (features['volatility'] < features['volatility'].quantile(0.7))
        
        # Bear market: negative trend and high volatility
        bear_mask = (features['return_20'] < -0.001) & (features['volatility'] > features['volatility'].quantile(0.5))
        
        regimes[bull_mask] = 2
        regimes[bear_mask] = 0
        
        return regimes
    
    def _sort_regimes(
        self,
        regimes: pd.Series,
        returns: pd.Series
    ) -> pd.Series:
        """Sort regimes by average return (0=bear, 1=sideways, 2=bull)."""
        # Calculate average return for each regime
        regime_returns = {}
        for regime in regimes.unique():
            mask = regimes == regime
            regime_returns[regime] = returns[mask].mean()
        
        # Sort regimes by return
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
        
        # Create mapping (old regime -> new regime)
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        
        # Apply mapping
        return regimes.map(regime_mapping)
    
    def get_regime_name(self, regime: int) -> str:
        """
        Get human-readable name for a regime.
        
        Parameters
        ----------
        regime : int
            Regime label (0, 1, 2).
            
        Returns
        -------
        str
            Regime name ("Bear", "Sideways", "Bull").
        """
        return self.regime_labels.get(regime, f"Regime {regime}")
