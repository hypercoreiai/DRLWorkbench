#!/usr/bin/env python3
"""
Test script for model training and prediction.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import SklearnForecaster
from src.utils.logging import setup_logging

def create_synthetic_sequences(n_samples=500, seq_len=10, n_features=5):
    """Create synthetic time series sequences."""
    np.random.seed(42)
    
    # Generate sequences
    X = np.random.randn(n_samples, seq_len, n_features)
    
    # Create target as a simple function of inputs
    # y = mean of last timestep + some noise
    y = X[:, -1, :].mean(axis=1) + np.random.randn(n_samples) * 0.1
    
    return X, y

def test_sklearn_forecaster():
    """Test sklearn-based forecaster."""
    logger = setup_logging()
    logger.info("Testing SklearnForecaster...")
    
    # Create synthetic data
    X, y = create_synthetic_sequences(n_samples=500, seq_len=10, n_features=5)
    
    # Split train/val/test
    n_train = 350
    n_val = 75
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    logger.info(f"Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    # Test different model types
    for model_type in ['ridge', 'rf', 'gbm']:
        logger.info(f"\n--- Testing {model_type} ---")
        
        # Build model
        model = SklearnForecaster(model_type=model_type)
        
        config = {}
        if model_type == 'ridge':
            config = {'alpha': 1.0}
        elif model_type in ['rf', 'gbm']:
            config = {'n_estimators': 50, 'max_depth': 5}
            if model_type == 'gbm':
                config['learning_rate'] = 0.1
        
        model.build(input_shape=(X_train.shape[1], X_train.shape[2]), config=config)
        
        # Train
        trained_model, history = model.fit(
            X_train, y_train,
            X_val, y_val,
            config=config
        )
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        
        logger.info(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}")
        logger.info(f"Hyperparams: {model.get_hyperparams()}")
        
        # Basic assertions
        assert y_pred.shape == y_test.shape, "Prediction shape mismatch"
        assert mse < 10.0, f"MSE too high for {model_type}: {mse}"
        assert len(history['val_loss']) > 0, "No validation history"
        
        logger.info(f"✓ {model_type} test passed!")
    
    return True

if __name__ == "__main__":
    try:
        success = test_sklearn_forecaster()
        print("\n✓ All model tests passed!")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
