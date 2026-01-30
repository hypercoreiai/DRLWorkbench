"""
Volatility Forecasting Data Pipeline
Prepares data for volatility modeling with realized volatility calculations
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.pipeline import DataBundle
from src.models.volatility_forecasting import RealizedVolatility


class VolatilityDataBundle(DataBundle):
    """
    Extended data bundle for volatility forecasting.
    Contains returns, realized volatility, and features.
    """
    
    def __init__(self):
        super().__init__()
        self.realized_vol_train: Optional[pd.DataFrame] = None
        self.realized_vol_test: Optional[pd.DataFrame] = None
        self.returns_train: Optional[pd.DataFrame] = None
        self.returns_test: Optional[pd.DataFrame] = None
        self.features_train: Optional[pd.DataFrame] = None
        self.features_test: Optional[pd.DataFrame] = None


class VolatilityPipeline:
    """
    Volatility forecasting data pipeline.
    
    Steps:
    1. Load multi-asset OHLCV data
    2. Calculate returns
    3. Calculate realized volatility (multiple estimators)
    4. Create volatility features
    5. Prepare sequences for forecasting
    6. Split train/test
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bundle = VolatilityDataBundle()
        
    def run(self) -> VolatilityDataBundle:
        """
        Execute full pipeline.
        
        Returns:
            VolatilityDataBundle with all data prepared.
        """
        # Load tickers
        tickers = self._load_tickers()
        if not tickers:
            self.bundle.error_log.append("No tickers found")
            return self.bundle
        
        # Load OHLCV data
        ohlcv_data = self._load_ohlcv_data(tickers)
        if not ohlcv_data:
            self.bundle.error_log.append("Failed to load OHLCV data")
            return self.bundle
        
        # Calculate returns
        returns_df = self._calculate_returns(ohlcv_data)
        
        # Calculate realized volatility
        realized_vol_df = self._calculate_realized_volatility(ohlcv_data)
        
        # Create features
        features_df = self._create_volatility_features(returns_df, realized_vol_df, ohlcv_data)
        
        # Align data
        returns_df, realized_vol_df, features_df = self._align_data(
            returns_df, realized_vol_df, features_df
        )
        
        # Split train/test
        self._split_train_test(returns_df, realized_vol_df, features_df)
        
        # Store metadata
        self.bundle.metadata['tickers'] = tickers
        self.bundle.metadata['n_assets'] = len(tickers)
        self.bundle.metadata['config'] = self.config
        self.bundle.metadata['vol_estimators'] = self.config.get('volatility', {}).get(
            'estimators', ['standard', 'parkinson', 'garman_klass']
        )
        
        return self.bundle
    
    def _load_tickers(self) -> List[str]:
        """Load ticker list from config or symbols file."""
        tickers = self.config.get('data', {}).get('tickers', [])
        
        if tickers:
            return tickers
        
        symbols_path = self.config.get('data', {}).get('symbols_file', 'src/symbols/portfolio')
        
        try:
            path = Path(symbols_path)
            if path.exists():
                with open(path, 'r') as f:
                    tickers = [line.strip() for line in f if line.strip()]
                return tickers
        except Exception as e:
            self.bundle.error_log.append(f"Failed to load symbols file: {e}")
        
        return []
    
    def _load_ohlcv_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data for all tickers."""
        from src.ohlcv.yfinance_ohlcv import YFinanceOHLCV
        
        yf = YFinanceOHLCV()
        period = self.config.get('data', {}).get('period', '2y')
        
        try:
            results = yf.get(tickers, period)
            valid_results = {k: v for k, v in results.items() if not v.empty}
            
            if not valid_results:
                self.bundle.error_log.append("No valid data returned from YFinance")
                
            return valid_results
            
        except Exception as e:
            self.bundle.error_log.append(f"Data download failed: {e}")
            return {}
    
    def _calculate_returns(self, ohlcv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate returns for all assets."""
        returns_dict = {}
        
        for ticker, df in ohlcv_data.items():
            if 'close' in df.columns:
                returns = df['close'].pct_change()
                returns_dict[ticker] = returns
        
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.iloc[1:]  # Drop first NaN
        
        return returns_df
    
    def _calculate_realized_volatility(
        self,
        ohlcv_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate realized volatility using multiple estimators.
        
        Returns:
            DataFrame with columns = (ticker, estimator) multi-index
        """
        estimators = self.config.get('volatility', {}).get(
            'estimators', ['standard', 'parkinson', 'garman_klass']
        )
        window = self.config.get('volatility', {}).get('window', 20)
        
        rv_dict = {}
        
        for ticker, df in ohlcv_data.items():
            ticker_rv = {}
            
            # Standard (close-to-close)
            if 'standard' in estimators and 'close' in df.columns:
                returns = df['close'].pct_change()
                ticker_rv['standard'] = RealizedVolatility.standard(returns, window)
            
            # Parkinson (high-low)
            if 'parkinson' in estimators and 'high' in df.columns and 'low' in df.columns:
                ticker_rv['parkinson'] = RealizedVolatility.parkinson(
                    df['high'], df['low'], window
                )
            
            # Garman-Klass (OHLC)
            if 'garman_klass' in estimators:
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    ticker_rv['garman_klass'] = RealizedVolatility.garman_klass(
                        df['open'], df['high'], df['low'], df['close'], window
                    )
            
            # Rogers-Satchell
            if 'rogers_satchell' in estimators:
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    ticker_rv['rogers_satchell'] = RealizedVolatility.rogers_satchell(
                        df['open'], df['high'], df['low'], df['close'], window
                    )
            
            # Yang-Zhang
            if 'yang_zhang' in estimators:
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    ticker_rv['yang_zhang'] = RealizedVolatility.yang_zhang(
                        df['open'], df['high'], df['low'], df['close'], window
                    )
            
            # Store as multi-column DataFrame
            for est_name, vol_series in ticker_rv.items():
                rv_dict[f"{ticker}_{est_name}"] = vol_series
        
        return pd.DataFrame(rv_dict)
    
    def _create_volatility_features(
        self,
        returns_df: pd.DataFrame,
        realized_vol_df: pd.DataFrame,
        ohlcv_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Create features for volatility forecasting.
        
        Features include:
        - Lagged realized volatility
        - Squared returns
        - Absolute returns
        - Volatility of volatility
        - Range-based measures
        """
        features_dict = {}
        
        for ticker in returns_df.columns:
            # Get relevant data
            returns = returns_df[ticker]
            
            # Lagged volatility features
            for est_name in ['standard', 'parkinson', 'garman_klass']:
                col_name = f"{ticker}_{est_name}"
                if col_name in realized_vol_df.columns:
                    vol_series = realized_vol_df[col_name]
                    
                    # Lags
                    features_dict[f"{col_name}_lag1"] = vol_series.shift(1)
                    features_dict[f"{col_name}_lag5"] = vol_series.shift(5)
                    features_dict[f"{col_name}_lag22"] = vol_series.shift(22)
                    
                    # HAR components (daily, weekly, monthly averages)
                    features_dict[f"{col_name}_har_daily"] = vol_series.rolling(1).mean()
                    features_dict[f"{col_name}_har_weekly"] = vol_series.rolling(5).mean()
                    features_dict[f"{col_name}_har_monthly"] = vol_series.rolling(22).mean()
                    
                    # Volatility of volatility
                    features_dict[f"{col_name}_volvol"] = vol_series.rolling(20).std()
            
            # Return-based features
            features_dict[f"{ticker}_return"] = returns
            features_dict[f"{ticker}_return_sq"] = returns ** 2
            features_dict[f"{ticker}_return_abs"] = np.abs(returns)
            features_dict[f"{ticker}_return_lag1"] = returns.shift(1)
            features_dict[f"{ticker}_return_sq_lag1"] = (returns ** 2).shift(1)
            
            # Rolling measures
            features_dict[f"{ticker}_vol_5d"] = returns.rolling(5).std() * np.sqrt(252)
            features_dict[f"{ticker}_vol_22d"] = returns.rolling(22).std() * np.sqrt(252)
            features_dict[f"{ticker}_vol_63d"] = returns.rolling(63).std() * np.sqrt(252)
            
            # Skewness and kurtosis
            features_dict[f"{ticker}_skew_22d"] = returns.rolling(22).skew()
            features_dict[f"{ticker}_kurt_22d"] = returns.rolling(22).kurt()
            
            # Range-based (if OHLC available)
            if ticker in ohlcv_data:
                df = ohlcv_data[ticker]
                if 'high' in df.columns and 'low' in df.columns:
                    hl_range = (df['high'] - df['low']) / df['low']
                    features_dict[f"{ticker}_hl_range"] = hl_range
                    features_dict[f"{ticker}_hl_range_ma"] = hl_range.rolling(10).mean()
        
        return pd.DataFrame(features_dict)
    
    def _align_data(
        self,
        returns_df: pd.DataFrame,
        realized_vol_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Align all DataFrames to have the same index."""
        # Get common index
        common_idx = returns_df.index.intersection(
            realized_vol_df.index
        ).intersection(features_df.index)
        
        # Reindex all
        returns_df = returns_df.loc[common_idx]
        realized_vol_df = realized_vol_df.loc[common_idx]
        features_df = features_df.loc[common_idx]
        
        # Drop any remaining NaN rows
        valid_idx = returns_df.dropna().index.intersection(
            realized_vol_df.dropna().index
        ).intersection(features_df.dropna().index)
        
        returns_df = returns_df.loc[valid_idx]
        realized_vol_df = realized_vol_df.loc[valid_idx]
        features_df = features_df.loc[valid_idx]
        
        return returns_df, realized_vol_df, features_df
    
    def _split_train_test(
        self,
        returns_df: pd.DataFrame,
        realized_vol_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> None:
        """Split data into train/test sets (temporal split)."""
        test_ratio = self.config.get('data', {}).get('test_ratio', 0.2)
        
        n = len(returns_df)
        split_idx = int(n * (1 - test_ratio))
        
        # Split
        self.bundle.returns_train = returns_df.iloc[:split_idx]
        self.bundle.returns_test = returns_df.iloc[split_idx:]
        
        self.bundle.realized_vol_train = realized_vol_df.iloc[:split_idx]
        self.bundle.realized_vol_test = realized_vol_df.iloc[split_idx:]
        
        self.bundle.features_train = features_df.iloc[:split_idx]
        self.bundle.features_test = features_df.iloc[split_idx:]
        
        # Store in legacy fields for compatibility
        self.bundle.X_train = features_df.iloc[:split_idx]
        self.bundle.X_test = features_df.iloc[split_idx:]
        self.bundle.y_train = realized_vol_df.iloc[:split_idx]
        self.bundle.y_test = realized_vol_df.iloc[split_idx:]


def create_vol_sequences(
    features: pd.DataFrame,
    target_vol: pd.Series,
    seq_length: int = 22,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for volatility forecasting.
    
    Args:
        features: DataFrame with features
        target_vol: Series of target volatility
        seq_length: Length of input sequences
        forecast_horizon: Number of steps ahead to forecast
    
    Returns:
        (X, y) tuple where:
            X: (N, seq_length, n_features)
            y: (N, forecast_horizon)
    """
    features_values = features.values
    target_values = target_vol.values
    
    X, y = [], []
    
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        # Input sequence
        X.append(features_values[i:i+seq_length])
        
        # Target (next forecast_horizon steps)
        y.append(target_values[i+seq_length:i+seq_length+forecast_horizon])
    
    return np.array(X), np.array(y)
