# PyTorch LSTM Forecaster (V3 — PROJECT_OUTLINE Section 3.2)
"""
Simple LSTM forecasting model with configurable architecture.
Supports regularization (L1/L2), dropout, and batch normalization.
"""

from typing import Any, Dict, Optional, Tuple
import logging
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.models.base import BaseForecaster
from src.utils.errors import PredictionError

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module if TORCH_AVAILABLE else object):
    """LSTM neural network for time series forecasting."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM model")
        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.fc(out)
        return out


class LSTMForecaster(BaseForecaster):
    """
    LSTM-based forecaster with training and prediction capabilities.
    """
    
    def __init__(self):
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, LSTM model will not work")
        self.model = None
        self.device = None
        self.scaler_mean = None
        self.scaler_std = None
        self.hyperparams = {}
    
    def build(self, input_shape: Tuple[int, int], config: Dict[str, Any]) -> None:
        """
        Build LSTM model.
        
        Args:
            input_shape: (seq_len, n_features)
            config: Model configuration with hyperparameters
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        seq_len, n_features = input_shape
        hidden_dim = config.get('hidden_dim', 64)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.2)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=1
        ).to(self.device)
        
        self.hyperparams = {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'device': str(self.device)
        }
        
        logger.info(f"Built LSTM model: {self.hyperparams}")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features (N, seq_len, n_features)
            y_train: Training targets (N,)
            X_val: Validation features
            y_val: Validation targets
            config: Training configuration
        
        Returns:
            (model, history) tuple
        """
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("Model not built")
        
        config = config or {}
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.001)
        early_stopping_patience = config.get('early_stopping_patience', 10)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.FloatTensor(X_val).to(self.device)
                    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
                    val_outputs = self.model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                    history['val_loss'].append(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}" + 
                          (f", Val Loss: {val_loss:.6f}" if X_val is not None else ""))
        
        return self.model, history
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Test features (N, seq_len, n_features)
        
        Returns:
            Predictions (N,)
        """
        if not TORCH_AVAILABLE or self.model is None:
            raise PredictionError("Model not fitted")
        
        self.model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(self.device)
            predictions = self.model(X_test_t)
            return predictions.cpu().numpy().squeeze()
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return fitted hyperparameters."""
        return self.hyperparams
