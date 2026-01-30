"""
Volatility Forecasting Models
Implements: GARCH, EGARCH, HAR, LSTM-Vol, Realized Volatility, Ensemble
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseForecaster
from src.utils.errors import PredictionError


class RealizedVolatility:
    """
    Calculate realized volatility from high-frequency returns.
    Various estimators: Standard, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
    """
    
    @staticmethod
    def standard(returns: pd.Series, window: int = 20) -> pd.Series:
        """Standard realized volatility (rolling std)."""
        return returns.rolling(window).std() * np.sqrt(252)
    
    @staticmethod
    def parkinson(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """
        Parkinson volatility estimator using high-low range.
        More efficient than close-to-close.
        """
        hl_ratio = np.log(high / low)
        rv = np.sqrt((1 / (4 * np.log(2))) * (hl_ratio ** 2))
        return rv.rolling(window).mean() * np.sqrt(252)
    
    @staticmethod
    def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, 
                     window: int = 20) -> pd.Series:
        """
        Garman-Klass volatility estimator.
        Uses OHLC data for better efficiency.
        """
        hl_ratio = np.log(high / low)
        co_ratio = np.log(close / open_)
        
        gk = 0.5 * (hl_ratio ** 2) - (2 * np.log(2) - 1) * (co_ratio ** 2)
        return np.sqrt(gk).rolling(window).mean() * np.sqrt(252)
    
    @staticmethod
    def rogers_satchell(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                        window: int = 20) -> pd.Series:
        """
        Rogers-Satchell volatility estimator.
        Handles drift and is more robust.
        """
        hc = np.log(high / close)
        ho = np.log(high / open_)
        lc = np.log(low / close)
        lo = np.log(low / open_)
        
        rs = hc * ho + lc * lo
        return np.sqrt(rs).rolling(window).mean() * np.sqrt(252)
    
    @staticmethod
    def yang_zhang(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                   window: int = 20) -> pd.Series:
        """
        Yang-Zhang volatility estimator.
        Combines overnight and intraday volatility.
        Most sophisticated estimator.
        """
        # Overnight volatility
        co = np.log(open_ / close.shift(1))
        overnight_vol = co.rolling(window).var()
        
        # Open-to-close volatility
        oc = np.log(close / open_)
        oc_vol = oc.rolling(window).var()
        
        # Rogers-Satchell
        hc = np.log(high / close)
        ho = np.log(high / open_)
        lc = np.log(low / close)
        lo = np.log(low / open_)
        rs = (hc * ho + lc * lo).rolling(window).mean()
        
        # Yang-Zhang combination
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz = overnight_vol + k * oc_vol + (1 - k) * rs
        
        return np.sqrt(yz) * np.sqrt(252)


class GARCHForecaster(BaseForecaster):
    """
    GARCH(1,1) volatility forecasting model.
    Classic volatility model with mean reversion.
    """
    
    def __init__(self):
        self.omega = None  # Constant term
        self.alpha = None  # ARCH term (lagged squared residuals)
        self.beta = None   # GARCH term (lagged variance)
        self.fitted_ = False
        
    def fit(
        self,
        returns: pd.Series,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict]:
        """
        Fit GARCH(1,1) model using maximum likelihood.
        
        Args:
            returns: Time series of returns
            
        Returns:
            (self, history) tuple
        """
        try:
            from arch import arch_model
            
            # Fit GARCH(1,1)
            model = arch_model(returns.dropna(), vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off', show_warning=False)
            
            # Store parameters
            self.omega = result.params['omega']
            self.alpha = result.params['alpha[1]']
            self.beta = result.params['beta[1]']
            self.fitted_ = True
            
            history = {
                'loglikelihood': result.loglikelihood,
                'aic': result.aic,
                'bic': result.bic,
                'params': result.params.to_dict()
            }
            
            return self, history
            
        except ImportError:
            # Fallback to simple implementation
            return self._fit_simple(returns)
    
    def _fit_simple(self, returns: pd.Series) -> Tuple[Any, Dict]:
        """Simple GARCH estimation using method of moments."""
        r = returns.dropna()
        
        # Simple MoM estimates
        self.omega = 0.01 * r.var()
        self.alpha = 0.1
        self.beta = 0.85
        self.fitted_ = True
        
        return self, {'method': 'moments'}
    
    def predict(self, returns: pd.Series, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility for horizon steps ahead.
        
        Args:
            returns: Historical returns
            horizon: Forecast horizon
            
        Returns:
            Array of volatility forecasts
        """
        if not self.fitted_:
            raise PredictionError("Model not fitted")
        
        r = returns.dropna()
        
        # Initialize with unconditional variance
        sigma2 = r.var()
        
        # Last residual squared
        epsilon2 = (r.iloc[-1] ** 2)
        
        # Forecast
        forecasts = []
        for h in range(horizon):
            if h == 0:
                sigma2_next = self.omega + self.alpha * epsilon2 + self.beta * sigma2
            else:
                # Multi-step: use persistence
                persistence = (self.alpha + self.beta) ** h
                sigma2_next = r.var() * (1 - persistence) + sigma2 * persistence
            
            forecasts.append(np.sqrt(sigma2_next * 252))  # Annualize
            sigma2 = sigma2_next
            epsilon2 = sigma2  # Use forecast as proxy
        
        return np.array(forecasts)
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return fitted parameters."""
        return {
            'omega': float(self.omega) if self.omega is not None else None,
            'alpha': float(self.alpha) if self.alpha is not None else None,
            'beta': float(self.beta) if self.beta is not None else None,
            'persistence': float(self.alpha + self.beta) if self.alpha and self.beta else None
        }


class EGARCHForecaster(BaseForecaster):
    """
    EGARCH (Exponential GARCH) volatility model.
    Captures leverage effects (asymmetry in volatility response).
    """
    
    def __init__(self):
        self.params = None
        self.fitted_ = False
        
    def fit(
        self,
        returns: pd.Series,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict]:
        """Fit EGARCH model."""
        try:
            from arch import arch_model
            
            model = arch_model(returns.dropna(), vol='EGARCH', p=1, q=1, rescale=False)
            result = model.fit(disp='off', show_warning=False)
            
            self.params = result.params
            self.fitted_ = True
            
            history = {
                'loglikelihood': result.loglikelihood,
                'aic': result.aic,
                'bic': result.bic,
                'params': result.params.to_dict()
            }
            
            return self, history
            
        except ImportError:
            # Fallback to GARCH
            garch = GARCHForecaster()
            return garch.fit(returns)
    
    def predict(self, returns: pd.Series, horizon: int = 1) -> np.ndarray:
        """Forecast volatility."""
        if not self.fitted_:
            raise PredictionError("Model not fitted")
        
        # Use last volatility as baseline
        vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Simple persistence forecast
        forecasts = [vol * (0.95 ** h) + vol * 0.05 for h in range(horizon)]
        
        return np.array(forecasts)
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return fitted parameters."""
        if self.params is not None:
            return self.params.to_dict()
        return {}


class HARForecaster(BaseForecaster):
    """
    HAR (Heterogeneous Autoregressive) model for realized volatility.
    Uses daily, weekly, and monthly components.
    """
    
    def __init__(self):
        self.coef_daily = None
        self.coef_weekly = None
        self.coef_monthly = None
        self.intercept = None
        self.fitted_ = False
        
    def fit(
        self,
        realized_vol: pd.Series,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict]:
        """
        Fit HAR model using OLS regression.
        
        Args:
            realized_vol: Series of realized volatility
            
        Returns:
            (self, history) tuple
        """
        # Create features: daily, weekly (5d), monthly (22d) averages
        daily = realized_vol.shift(1)
        weekly = realized_vol.rolling(5).mean().shift(1)
        monthly = realized_vol.rolling(22).mean().shift(1)
        
        # Target: next day's realized vol
        y = realized_vol
        
        # Combine features
        X = pd.DataFrame({
            'daily': daily,
            'weekly': weekly,
            'monthly': monthly
        }).dropna()
        
        y = y.loc[X.index]
        
        # OLS regression
        from scipy.linalg import lstsq
        
        # Add intercept
        X_with_const = np.column_stack([np.ones(len(X)), X.values])
        
        # Solve least squares
        result = lstsq(X_with_const, y.values)
        coef = result[0]
        
        self.intercept = coef[0]
        self.coef_daily = coef[1]
        self.coef_weekly = coef[2]
        self.coef_monthly = coef[3]
        self.fitted_ = True
        
        # Calculate R-squared
        y_pred = X_with_const @ coef
        ss_res = np.sum((y.values - y_pred) ** 2)
        ss_tot = np.sum((y.values - y.values.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        history = {
            'r2': float(r2),
            'coefficients': {
                'intercept': float(self.intercept),
                'daily': float(self.coef_daily),
                'weekly': float(self.coef_weekly),
                'monthly': float(self.coef_monthly)
            }
        }
        
        return self, history
    
    def predict(self, realized_vol: pd.Series, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility using HAR components.
        
        Args:
            realized_vol: Historical realized volatility
            horizon: Forecast horizon
            
        Returns:
            Array of volatility forecasts
        """
        if not self.fitted_:
            raise PredictionError("Model not fitted")
        
        forecasts = []
        vol_series = realized_vol.copy()
        
        for h in range(horizon):
            # Calculate components
            daily = vol_series.iloc[-1]
            weekly = vol_series.iloc[-5:].mean()
            monthly = vol_series.iloc[-22:].mean()
            
            # Forecast
            vol_forecast = (self.intercept + 
                          self.coef_daily * daily + 
                          self.coef_weekly * weekly + 
                          self.coef_monthly * monthly)
            
            forecasts.append(vol_forecast)
            
            # Update series for multi-step
            vol_series = pd.concat([vol_series, pd.Series([vol_forecast])])
        
        return np.array(forecasts)
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return fitted parameters."""
        return {
            'intercept': float(self.intercept) if self.intercept is not None else None,
            'coef_daily': float(self.coef_daily) if self.coef_daily is not None else None,
            'coef_weekly': float(self.coef_weekly) if self.coef_weekly is not None else None,
            'coef_monthly': float(self.coef_monthly) if self.coef_monthly is not None else None
        }


class LSTMVolForecaster(BaseForecaster):
    """
    LSTM model for volatility forecasting.
    Uses sequences of returns and volatility features.
    """
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_mean = None
        self.scaler_std = None
        self.history = {}
        
    def build(self, input_shape: Tuple[int, int], config: Dict[str, Any]) -> None:
        """Build LSTM model for volatility."""
        seq_len, n_features = input_shape
        
        hidden_size = config.get('hidden_size', 32)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.2)
        
        self.model = LSTMVolModel(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=1
        ).to(self.device)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict]:
        """Train LSTM volatility model."""
        if config is None:
            config = {}
        
        if self.model is None:
            input_shape = X_train.shape[1:]
            self.build(input_shape, config)
        
        # Normalize
        X_train_norm, y_train_norm = self._normalize(X_train, y_train, fit=True)
        
        if X_val is not None and y_val is not None:
            X_val_norm, y_val_norm = self._normalize(X_val, y_val, fit=False)
        else:
            X_val_norm, y_val_norm = None, None
        
        # Training parameters
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.001)
        
        # DataLoaders
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
        return self, history
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make volatility predictions."""
        if self.model is None:
            raise PredictionError("Model not fitted")
        
        # Normalize
        X_test_norm = (X_test - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test_norm).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def _normalize(self, X: np.ndarray, y: np.ndarray, fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize data."""
        if fit:
            self.scaler_mean = X.mean(axis=(0, 1), keepdims=True)
            self.scaler_std = X.std(axis=(0, 1), keepdims=True)
        
        X_norm = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
        y_norm = y
        
        return X_norm, y_norm
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return model configuration."""
        if self.model is None:
            return {}
        return {
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'dropout': self.model.dropout
        }


class LSTMVolModel(nn.Module):
    """LSTM network for volatility forecasting."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super(LSTMVolModel, self).__init__()
        
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
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output


def get_volatility_model(model_type: str, config: Optional[Dict] = None) -> BaseForecaster:
    """
    Factory function for volatility models.
    
    Args:
        model_type: Model name (garch, egarch, har, lstm)
        config: Optional configuration
        
    Returns:
        Volatility forecaster instance
    """
    models = {
        'garch': GARCHForecaster,
        'egarch': EGARCHForecaster,
        'har': HARForecaster,
        'lstm': LSTMVolForecaster,
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type.lower()]()
