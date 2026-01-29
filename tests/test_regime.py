# Unit tests for regime detection and regime-aware metrics (V3 — PROJECT_OUTLINE Section 12)

import numpy as np
import pandas as pd
import pytest

from src.backtest.regime import detect_regimes, compute_regime_metrics


class TestDetectRegimes:
    def test_volatility_regimes(self):
        np.random.seed(42)
        r = pd.Series(np.random.randn(200) * 0.01)
        labels = detect_regimes(r, method="volatility", window=21)
        assert len(labels) == len(r)
        assert labels.min() >= 0
        assert labels.max() <= 2

    def test_return_regimes(self):
        np.random.seed(42)
        r = pd.Series(np.random.randn(200) * 0.01)
        labels = detect_regimes(r, method="return", window=21)
        assert len(labels) == len(r)


class TestComputeRegimeMetrics:
    def test_compute_regime_metrics(self):
        np.random.seed(42)
        r = pd.Series(np.random.randn(200) * 0.01)
        labels = np.random.randint(0, 3, size=200)
        df = compute_regime_metrics(r, labels)
        assert isinstance(df, pd.DataFrame)
        assert "regime" in df.columns
        assert "sharpe" in df.columns
        assert "max_drawdown" in df.columns
