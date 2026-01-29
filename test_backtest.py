#!/usr/bin/env python3
"""
Test script for walk-forward backtesting.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.pipeline import DataPipeline, DataBundle
from src.backtest.walker import WalkForwardBacktester
from src.models import SklearnForecaster
from src.utils.logging import setup_logging

def create_synthetic_ohlcv(n_days=500, seed=42):
    """Create synthetic OHLCV data for testing."""
    np.random.seed(seed)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    
    # Generate price series with some trend and noise
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)
    close = base_price * np.cumprod(1 + returns)
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_ = close * (1 + np.random.normal(0, 0.005, n_days))
    volume = np.random.lognormal(15, 1, n_days)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df

def test_walk_forward_backtest():
    """Test walk-forward backtesting."""
    logger = setup_logging()
    logger.info("Testing Walk-Forward Backtester...")
    
    # Create config
    config = {
        "data": {
            "tickers": ["SYNTH1"],
            "time_step": 10,
            "look_ahead": 1,
            "test_size": 0.2,
            "feature_selection": {
                "method": "correlation",
                "threshold": 0.3
            },
            "validation": {
                "check_stationarity": False,
                "check_leakage": True,
                "outlier_method": "iqr",
                "missing_threshold": 0.05
            }
        },
        "backtest": {
            "train_window": 100,  # Smaller for testing
            "test_window": 30,
            "rebalance_freq": 15,
            "transaction_cost": 0.001,
            "bid_ask_spread": 0.001,
            "regime_detection": {
                "method": "volatility",
                "periods": 20
            }
        },
        "models": [{
            "type": "ridge",
            "alpha": 1.0
        }],
        "display": {
            "run_id": "test_backtest_001"
        }
    }
    
    # Create pipeline with mocked data download
    pipeline = DataPipeline(config)
    
    def mock_download():
        return {"SYNTH1": create_synthetic_ohlcv()}
    
    pipeline._download_ohlcv = mock_download
    
    # Run pipeline to get data
    logger.info("Running data pipeline...")
    bundle = pipeline.run()
    
    logger.info(f"Data prepared: {bundle.X_train.shape[0]} train, {bundle.X_test.shape[0]} test")
    
    # Create backtester
    backtester = WalkForwardBacktester(bundle, config)
    
    # Define model builder
    def build_model():
        model = SklearnForecaster(model_type='ridge')
        return model
    
    # Run backtest
    logger.info("Running walk-forward backtest...")
    report = backtester.run(model_builder=build_model)
    
    # Check results
    logger.info(f"Backtest completed!")
    logger.info(f"Number of steps: {report['metrics']['n_steps']}")
    logger.info(f"Total return: {report['metrics']['total_return']:.4f}")
    logger.info(f"Sharpe ratio: {report['metrics']['sharpe']:.4f}")
    logger.info(f"Max drawdown: {report['metrics']['max_drawdown']:.4f}")
    logger.info(f"Returns count: {len(report['returns'])}")
    logger.info(f"Predictions count: {len(report['predictions'])}")
    
    if report['regime_info']:
        logger.info(f"Regime analysis: {len(report['regime_info'])} regimes detected")
        for regime in report['regime_info']:
            logger.info(f"  Regime {regime['regime']}: Sharpe={regime.get('sharpe', 'N/A'):.3f}")
    
    # Assertions
    assert report['metrics']['n_steps'] > 0, "No backtest steps completed"
    assert len(report['returns']) > 0, "No returns recorded"
    assert len(report['predictions']) > 0, "No predictions recorded"
    assert len(report['predictions']) == len(report['actuals']), "Prediction/actual mismatch"
    assert len(report['cumulative_returns']) > 0, "No cumulative returns"
    
    logger.info("✓ All backtest assertions passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_walk_forward_backtest()
        print("\n✓ Walk-forward backtest test passed!")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
