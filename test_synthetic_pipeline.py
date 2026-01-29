#!/usr/bin/env python3
"""
Test script for the data pipeline implementation using synthetic data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.pipeline import DataPipeline, DataBundle
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

def test_pipeline_with_synthetic_data():
    """Test the data pipeline with synthetic data."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting pipeline test with synthetic data...")
    
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
        "display": {
            "run_id": "test_synthetic_001"
        }
    }
    
    # Create pipeline with mocked data download
    pipeline = DataPipeline(config)
    
    # Override the download method to use synthetic data
    def mock_download():
        return {"SYNTH1": create_synthetic_ohlcv()}
    
    pipeline._download_ohlcv = mock_download
    
    # Run pipeline
    try:
        bundle = pipeline.run()
        
        # Check results
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"X_train shape: {bundle.X_train.shape if bundle.X_train is not None else 'None'}")
        logger.info(f"y_train shape: {bundle.y_train.shape if bundle.y_train is not None else 'None'}")
        logger.info(f"X_test shape: {bundle.X_test.shape if bundle.X_test is not None else 'None'}")
        logger.info(f"y_test shape: {bundle.y_test.shape if bundle.y_test is not None else 'None'}")
        logger.info(f"Validation report: {bundle.validation_report}")
        logger.info(f"Metadata keys: {list(bundle.metadata.keys())}")
        
        if bundle.X_train is not None:
            logger.info(f"Number of features: {bundle.X_train.shape[2]}")
            logger.info(f"Training samples: {bundle.X_train.shape[0]}")
            logger.info(f"Test samples: {bundle.X_test.shape[0]}")
        
        if bundle.error_log:
            logger.warning(f"Errors encountered: {bundle.error_log}")
        
        # Validate shapes
        assert bundle.X_train is not None, "X_train should not be None"
        assert bundle.y_train is not None, "y_train should not be None"
        assert bundle.X_test is not None, "X_test should not be None"
        assert bundle.y_test is not None, "y_test should not be None"
        assert bundle.X_train.shape[0] > 0, "X_train should have samples"
        assert bundle.X_test.shape[0] > 0, "X_test should have samples"
        
        logger.info("✓ All assertions passed!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline_with_synthetic_data()
    sys.exit(0 if success else 1)
