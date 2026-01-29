"""Tests for regime detection module."""

import pytest
import pandas as pd
import numpy as np

from drlworkbench.regime import RegimeDetector


def test_regime_detector_initialization():
    """Test RegimeDetector initialization."""
    detector = RegimeDetector(method="kmeans", n_regimes=3)
    
    assert detector.method == "kmeans"
    assert detector.n_regimes == 3


def test_regime_detection_kmeans():
    """Test regime detection with K-Means."""
    np.random.seed(42)
    
    # Create synthetic returns with different regimes
    returns = pd.Series(
        np.concatenate([
            np.random.normal(0.02, 0.01, 100),  # Bull
            np.random.normal(-0.02, 0.03, 100),  # Bear
            np.random.normal(0.0, 0.015, 100),  # Sideways
        ]),
        index=pd.date_range('2020-01-01', periods=300)
    )
    
    detector = RegimeDetector(method="kmeans", n_regimes=3)
    regimes = detector.fit_predict(returns)
    
    assert len(regimes) == len(returns)
    assert set(regimes.unique()).issubset({0, 1, 2})


def test_regime_detection_rule_based():
    """Test regime detection with rule-based method."""
    np.random.seed(42)
    
    returns = pd.Series(
        np.random.normal(0.001, 0.02, 100),
        index=pd.date_range('2020-01-01', periods=100)
    )
    
    detector = RegimeDetector(method="rule_based", n_regimes=3)
    regimes = detector.fit_predict(returns)
    
    assert len(regimes) == len(returns)
    assert set(regimes.unique()).issubset({0, 1, 2})


def test_regime_detector_get_regime_name():
    """Test getting regime name."""
    detector = RegimeDetector()
    
    assert detector.get_regime_name(0) == "Bear"
    assert detector.get_regime_name(1) == "Sideways"
    assert detector.get_regime_name(2) == "Bull"
