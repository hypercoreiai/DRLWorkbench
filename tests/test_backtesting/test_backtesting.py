"""Tests for backtesting module."""

import pytest
import pandas as pd
import numpy as np

from drlworkbench.backtesting import BacktestEngine, Portfolio
from drlworkbench.utils.exceptions import BacktestError


def test_backtest_engine_initialization():
    """Test BacktestEngine initialization."""
    engine = BacktestEngine(
        initial_capital=50000.0,
        commission_rate=0.002,
        slippage_rate=0.001
    )
    
    assert engine.initial_capital == 50000.0
    assert engine.commission_rate == 0.002
    assert engine.slippage_rate == 0.001


def test_portfolio_initialization():
    """Test Portfolio initialization."""
    portfolio = Portfolio(initial_capital=100000.0)
    
    assert portfolio.initial_capital == 100000.0
    assert portfolio.cash == 100000.0
    assert len(portfolio.positions) == 0


def test_portfolio_reset():
    """Test Portfolio reset."""
    portfolio = Portfolio(initial_capital=100000.0)
    portfolio.cash = 50000.0
    portfolio.positions = {"AAPL": 10}
    
    portfolio.reset()
    
    assert portfolio.cash == 100000.0
    assert len(portfolio.positions) == 0


def test_portfolio_get_value():
    """Test Portfolio value calculation."""
    portfolio = Portfolio(initial_capital=100000.0)
    portfolio.cash = 50000.0
    portfolio.positions = {"AAPL": 10, "GOOGL": 5}
    
    prices = {"AAPL": 150.0, "GOOGL": 2800.0}
    value = portfolio.get_value(prices)
    
    expected = 50000.0 + (10 * 150.0) + (5 * 2800.0)
    assert value == expected


def test_backtest_with_invalid_data():
    """Test backtest with invalid data raises error."""
    engine = BacktestEngine()
    
    # Data without required columns
    data = pd.DataFrame({
        'price': [100, 101, 102],
    }, index=pd.date_range('2020-01-01', periods=3))
    
    def simple_strategy(data):
        return None
    
    with pytest.raises(BacktestError):
        engine.run(data, simple_strategy)
