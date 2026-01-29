# Unit tests for walk-forward backtest and transaction costs (V3 — PROJECT_OUTLINE Section 12)

import numpy as np
import pandas as pd
import pytest

from src.backtest import WalkForwardBacktester, PortfolioSimulator


class TestWalkForwardBacktester:
    def test_init(self):
        data = pd.DataFrame(np.random.randn(300, 3))
        config = {
            "backtest": {
                "train_window": 252,
                "test_window": 63,
                "rebalance_freq": 21,
            }
        }
        w = WalkForwardBacktester(data, config)
        assert w.train_window == 252
        assert w.test_window == 63
        assert w.rebalance_freq == 21

    def test_run_returns_report(self):
        data = pd.DataFrame(np.random.randn(300, 3))
        config = {"backtest": {"train_window": 252, "test_window": 63, "rebalance_freq": 21}}
        w = WalkForwardBacktester(data, config)
        report = w.run()
        assert "cumulative_returns" in report
        assert "metrics" in report
        assert "regime_info" in report


class TestPortfolioSimulator:
    def test_rebalance_costs(self):
        sim = PortfolioSimulator()
        old_w = np.array([0.5, 0.5])
        new_w = np.array([0.6, 0.4])
        prices = np.array([100.0, 200.0])
        out = sim.rebalance(old_w, new_w, prices, bid_ask_spread=0.001, commission=0.001)
        assert "turnover" in out
        assert "cost" in out
        assert out["turnover"] > 0
        assert out["cost"] > 0
