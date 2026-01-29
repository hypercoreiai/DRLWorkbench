"""
Simple example demonstrating DRLWorkbench usage.

This example shows how to:
1. Set up logging
2. Validate data
3. Detect market regimes
4. Run a basic backtest
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from drlworkbench import setup_logger, DataValidator, RegimeDetector, BacktestEngine

# Set up logging
logger = setup_logger("example", level=20)  # INFO level
logger.info("Starting DRLWorkbench example")

# Generate synthetic data
def generate_synthetic_data(n_days=252):
    """Generate synthetic price data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': prices * (1 + np.random.uniform(0.0, 0.02, n_days)),
        'low': prices * (1 + np.random.uniform(-0.02, 0.0, n_days)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)
    
    return data

# Generate data
logger.info("Generating synthetic price data...")
data = generate_synthetic_data(252)
logger.info(f"Generated {len(data)} days of data")

# Validate data
logger.info("Validating data quality...")
validator = DataValidator()
validation_results = validator.validate_all(data)

if validation_results['is_valid']:
    logger.info("✓ Data validation passed")
else:
    logger.warning(f"✗ Data validation issues: {validation_results['issues']}")

# Detect regimes
logger.info("Detecting market regimes...")
returns = data['close'].pct_change().dropna()
detector = RegimeDetector(method='kmeans', n_regimes=3)
regimes = detector.fit_predict(returns)

regime_counts = regimes.value_counts().sort_index()
logger.info("Regime distribution:")
for regime, count in regime_counts.items():
    regime_name = detector.get_regime_name(regime)
    logger.info(f"  {regime_name}: {count} days ({count/len(regimes)*100:.1f}%)")

# Test stationarity
logger.info("Testing returns stationarity...")
stationarity_result = validator.test_stationarity(returns)
if stationarity_result['is_stationary']:
    logger.info(f"✓ Returns are stationary (p-value: {stationarity_result['p_value']:.4f})")
else:
    logger.info(f"✗ Returns may not be stationary (p-value: {stationarity_result['p_value']:.4f})")

# Simple buy-and-hold strategy
def buy_and_hold_strategy(data, **params):
    """
    Simple buy-and-hold strategy (for demonstration).
    
    In practice, this would return trading signals.
    """
    # This is a placeholder - actual implementation would return signals
    return None

logger.info("\nExample completed successfully!")
logger.info("=" * 60)
logger.info("DRLWorkbench is ready for use.")
logger.info("Explore the documentation for more advanced features:")
logger.info("- Backtesting with walk-forward validation")
logger.info("- Hyperparameter optimization")
logger.info("- Ensemble models")
logger.info("- Professional reporting")
logger.info("=" * 60)
