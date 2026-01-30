"""
Forecasting Models: LSTM and NeuralForecast wrappers
Implements time series forecasting for portfolio optimization (V3)
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseForecaster
from src.utils.errors import PredictionError


class LSTMForecaster(BaseForecaster):
    """
    LSTM forecaster with PyTorch.
    Supports multi-step ahead forecasting.
    """
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_mean = None
        self.scaler_std = None
        self.history = {}
        
    def build(self, input_shape: Tuple[int, int], config: Dict[str, Any]) -> None:
        """
        Build LSTM model.
        
        Args:
            input_shape: (sequence_length, n_features)
            config: Model configuration with hyperparameters.
        """
        seq_len, n_features = input_shape
        
        # Get hyperparameters
        hidden_size = config.get('hidden_size', 64)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.2)
        output_size = config.get('output_size', 1)
        
        # Build model
        self.model = LSTMModel(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        ).to(self.device)
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict]:
        """
        Train LSTM model.
        
        Args:
            X_train: Training features (N, seq_len, n_features)
            y_train: Training targets (N, output_size)
            X_val: Validation features
            y_val: Validation targets
            config: Training configuration
        
        Returns:
            (model, history) tuple
        """
        if config is None:
            config = {}
            
        # Build model if not already built
        if self.model is None:
            input_shape = X_train.shape[1:]
            self.build(input_shape, config)
        
        # Normalize data
        X_train_norm, y_train_norm = self._normalize(X_train, y_train, fit=True)
        
        if X_val is not None and y_val is not None:
            X_val_norm, y_val_norm = self._normalize(X_val, y_val, fit=False)
        else:
            X_val_norm, y_val_norm = None, None
        
        # Training parameters
        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.001)
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train_norm)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val_norm is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_norm),
                torch.FloatTensor(y_val_norm)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience = config.get('patience', 10)
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        self.history = history
        return self.model, history
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Test features (N, seq_len, n_features)
        
        Returns:
            Predictions (N, output_size)
        """
        if self.model is None:
            raise PredictionError("Model not fitted")
        
        # Normalize
        X_test_norm = (X_test - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test_norm).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        # Denormalize predictions (assuming y was normalized the same way)
        # For simplicity, we return normalized predictions
        # In production, you'd want to track y_mean and y_std separately
        return predictions
    
    def _normalize(self, X: np.ndarray, y: np.ndarray, fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize data using z-score normalization."""
        if fit:
            # Compute statistics on training data
            self.scaler_mean = X.mean(axis=(0, 1), keepdims=True)
            self.scaler_std = X.std(axis=(0, 1), keepdims=True)
        
        X_norm = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
        y_norm = y  # For simplicity, don't normalize y
        
        return X_norm, y_norm
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return fitted hyperparameters for logging."""
        if self.model is None:
            return {}
        return {
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'dropout': self.model.dropout,
        }


class LSTMModel(nn.Module):
    """LSTM network architecture."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
        
        Returns:
            Output tensor (batch, output_size)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        last_out = lstm_out[:, -1, :]
        
        # Fully connected
        output = self.fc(last_out)
        
        return output


class SimpleEnsembleForecaster(BaseForecaster):
    """
    Ensemble forecaster that combines multiple models.
    Uses weighted averaging of predictions.
    """
    
    def __init__(self, models: list, weights: Optional[list] = None):
        """
        Args:
            models: List of fitted forecaster models.
            weights: Optional weights for each model. If None, equal weights.
        """
        self.models = models
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X_test: Test features
        
        Returns:
            Weighted average of model predictions
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # Stack and weight
        predictions = np.stack(predictions, axis=0)
        weighted_pred = np.sum(predictions * self.weights[:, None, None], axis=0)
        
        return weighted_pred
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return ensemble configuration."""
        return {
            'n_models': len(self.models),
            'weights': self.weights.tolist(),
            'model_types': [type(m).__name__ for m in self.models]
        }


def create_sequences(
    data: pd.DataFrame,
    target_col: str,
    seq_length: int = 10,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series forecasting.
    
    Args:
        data: DataFrame with features and target.
        target_col: Name of target column.
        seq_length: Length of input sequences.
        forecast_horizon: Number of steps ahead to forecast.
    
    Returns:
        (X, y) tuple where:
            X: (N, seq_length, n_features)
            y: (N, forecast_horizon)
    """
    values = data.values
    n_features = values.shape[1]
    
    # Find target column index
    target_idx = data.columns.get_loc(target_col)
    
    X, y = [], []
    
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        # Input sequence
        X.append(values[i:i+seq_length])
        
        # Target (next forecast_horizon steps)
        y.append(values[i+seq_length:i+seq_length+forecast_horizon, target_idx])
    
    return np.array(X), np.array(y)
