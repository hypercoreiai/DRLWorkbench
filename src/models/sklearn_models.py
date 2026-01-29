# Scikit-learn based forecasting models (simpler baselines)
"""
Simple forecasting models using scikit-learn for baseline comparisons.
Includes Linear Regression, Random Forest, and Gradient Boosting.
"""

from typing import Any, Dict, Optional, Tuple
import logging
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.models.base import BaseForecaster
from src.utils.errors import PredictionError

logger = logging.getLogger(__name__)


class SklearnForecaster(BaseForecaster):
    """
    Wrapper for sklearn regression models for time series forecasting.
    Flattens time series sequences into feature vectors.
    """
    
    def __init__(self, model_type: str = 'ridge'):
        """
        Initialize forecaster.
        
        Args:
            model_type: 'ridge', 'rf' (random forest), or 'gbm' (gradient boosting)
        """
        self.model_type = model_type
        self.model = None
        self.hyperparams = {}
    
    def build(self, input_shape: Tuple[int, int], config: Dict[str, Any]) -> None:
        """
        Build sklearn model.
        
        Args:
            input_shape: (seq_len, n_features) - will be flattened
            config: Model configuration
        """
        if self.model_type == 'ridge':
            alpha = config.get('alpha', 1.0)
            self.model = Ridge(alpha=alpha)
            self.hyperparams = {'model': 'Ridge', 'alpha': alpha}
        
        elif self.model_type == 'rf':
            n_estimators = config.get('n_estimators', 100)
            max_depth = config.get('max_depth', 10)
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            self.hyperparams = {
                'model': 'RandomForest',
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }
        
        elif self.model_type == 'gbm':
            n_estimators = config.get('n_estimators', 100)
            learning_rate = config.get('learning_rate', 0.1)
            max_depth = config.get('max_depth', 5)
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
            self.hyperparams = {
                'model': 'GradientBoosting',
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth
            }
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        logger.info(f"Built {self.hyperparams['model']} model: {self.hyperparams}")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train the model.
        
        Args:
            X_train: Training features (N, seq_len, n_features)
            y_train: Training targets (N,)
            X_val: Validation features
            y_val: Validation targets
            config: Training configuration (unused for sklearn)
        
        Returns:
            (model, history) tuple
        """
        if self.model is None:
            raise RuntimeError("Model not built")
        
        # Flatten sequences: (N, seq_len, n_features) -> (N, seq_len * n_features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Train
        self.model.fit(X_train_flat, y_train)
        
        # Compute training metrics
        y_train_pred = self.model.predict(X_train_flat)
        train_loss = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        history = {
            'loss': [train_loss],
            'mae': [train_mae]
        }
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_val_pred = self.model.predict(X_val_flat)
            val_loss = mean_squared_error(y_val, y_val_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            history['val_loss'] = [val_loss]
            history['val_mae'] = [val_mae]
            
            logger.info(
                f"Training complete - Train MSE: {train_loss:.6f}, "
                f"Val MSE: {val_loss:.6f}"
            )
        else:
            logger.info(f"Training complete - Train MSE: {train_loss:.6f}")
        
        return self.model, history
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Test features (N, seq_len, n_features)
        
        Returns:
            Predictions (N,)
        """
        if self.model is None:
            raise PredictionError("Model not fitted")
        
        # Flatten sequences
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_test_flat)
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return fitted hyperparameters."""
        return self.hyperparams
