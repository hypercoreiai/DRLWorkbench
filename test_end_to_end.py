#!/usr/bin/env python3
"""
End-to-end test of the complete pipeline using synthetic data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.run_pipeline import run_pipeline
from src.data.pipeline import DataPipeline
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

def test_end_to_end_pipeline():
    """Test the complete pipeline end-to-end."""
    logger = setup_logging()
    logger.info("Testing end-to-end pipeline...")
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp(prefix="drl_test_")
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create config file
        config_path = Path(temp_dir) / "test_config.yaml"
        config_content = """
data:
  tickers: ["SYNTH1"]
  time_step: 10
  look_ahead: 1
  test_size: 0.2
  feature_selection:
    method: "correlation"
    threshold: 0.3
  validation:
    check_stationarity: false
    check_leakage: true
    outlier_method: "iqr"
    missing_threshold: 0.05

backtest:
  train_window: 100
  test_window: 30
  rebalance_freq: 15
  transaction_cost: 0.001
  bid_ask_spread: 0.001
  regime_detection:
    method: "volatility"
    periods: 20

models:
  - type: "ridge"
    alpha: 1.0

display:
  run_id: "test_e2e_001"
"""
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Monkey-patch DataPipeline to use synthetic data
        original_download = DataPipeline._download_ohlcv
        
        def mock_download(self):
            return {"SYNTH1": create_synthetic_ohlcv()}
        
        DataPipeline._download_ohlcv = mock_download
        
        # Run pipeline
        logger.info("Running complete pipeline...")
        output_dir = Path(temp_dir) / "outputs"
        
        # Restore original method before running (so test doesn't break other tests)
        DataPipeline_download_backup = DataPipeline._download_ohlcv
        try:
            DataPipeline._download_ohlcv = mock_download
            run_pipeline(str(config_path), str(output_dir))
        finally:
            # Restore original method
            DataPipeline._download_ohlcv = DataPipeline_download_backup
        
        # Check outputs
        logger.info("Checking outputs...")
        
        # Check logs
        log_dir = output_dir / "logs"
        assert log_dir.exists(), "Log directory not created"
        log_files = list(log_dir.glob("*.log"))
        # Note: may be empty if using console-only logging
        logger.info(f"Found {len(log_files)} log files")
        
        # Check results
        results_dir = output_dir / "results"
        assert results_dir.exists(), "Results directory not created"
        
        report_file = results_dir / "test_e2e_001_report.json"
        assert report_file.exists(), "Report file not created"
        logger.info("✓ Report file created")
        
        summary_file = results_dir / "test_e2e_001_summary.txt"
        assert summary_file.exists(), "Summary file not created"
        logger.info("✓ Summary file created")
        
        # Check summary content
        with open(summary_file, 'r') as f:
            summary = f.read()
            assert "Sharpe ratio" in summary, "Summary missing Sharpe ratio"
            assert "Max drawdown" in summary, "Summary missing max drawdown"
            logger.info("✓ Summary contains expected metrics")
        
        # Check checkpoints
        checkpoint_files = list(output_dir.glob("checkpoint*.pkl"))
        assert len(checkpoint_files) >= 1, "No checkpoint files created"
        logger.info(f"✓ Found {len(checkpoint_files)} checkpoint files")
        
        logger.info("✓ All output checks passed!")
        
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    try:
        success = test_end_to_end_pipeline()
        print("\n✓ End-to-end pipeline test passed!")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
