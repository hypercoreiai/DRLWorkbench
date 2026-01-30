"""
Autoencoder Data Pipeline
Prepares market data for autoencoder training with comprehensive features
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.pipeline import DataBundle


class AutoencoderDataBundle(DataBundle):
    """Extended data bundle for autoencoder training."""
    
    def __init__(self):
        super().__init__()
        self.features_train: Optional[pd.DataFrame] = None
        self.features_test: Optional[pd.DataFrame] = None
        self.returns_train: Optional[pd.DataFrame] = None
        self.returns_test: Optional[pd.DataFrame] = None


class AutoencoderPipeline:
    """
    Data pipeline for market autoencoder.
    Creates comprehensive market state features from OHLCV data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bundle = AutoencoderDataBundle()
        
    def run(self) -> AutoencoderDataBundle:
        """Execute full pipeline."""
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
        
        # Create comprehensive features
        features_df = self._create_market_features(ohlcv_data, returns_df)
        
        # Align and split
        returns_df, features_df = self._align_data(returns_df, features_df)
        self._split_train_test(returns_df, features_df)
        
        # Store metadata
        self.bundle.metadata['tickers'] = tickers
        self.bundle.metadata['n_assets'] = len(tickers)
        self.bundle.metadata['n_features'] = features_df.shape[1]
        self.bundle.metadata['config'] = self.config
        
        return self.bundle
    
    def _load_tickers(self) -> List[str]:
        """Load ticker list."""
        tickers = self.config.get('data', {}).get('tickers', [])
        if tickers:
            return tickers
        
        symbols_path = self.config.get('data', {}).get('symbols_file', 'src/symbols/portfolio')
        try:
            path = Path(symbols_path)
            if path.exists():
                with open(path, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
        except Exception as e:
            self.bundle.error_log.append(f"Failed to load symbols: {e}")
        return []
    
    def _load_ohlcv_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data."""
        from src.ohlcv.yfinance_ohlcv import YFinanceOHLCV
        
        yf = YFinanceOHLCV()
        period = self.config.get('data', {}).get('period', '2y')
        
        try:
            results = yf.get(tickers, period)
            return {k: v for k, v in results.items() if not v.empty}
        except Exception as e:
            self.bundle.error_log.append(f"Data download failed: {e}")
            return {}
    
    def _calculate_returns(self, ohlcv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate returns."""
        returns_dict = {}
        for ticker, df in ohlcv_data.items():
            if 'close' in df.columns:
                returns_dict[ticker] = df['close'].pct_change()
        return pd.DataFrame(returns_dict).iloc[1:]
    
    def _create_market_features(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        returns_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create comprehensive market features."""
        features_dict = {}
        
        for ticker in returns_df.columns:
            if ticker not in ohlcv_data:
                continue
            
            df = ohlcv_data[ticker]
            returns = returns_df[ticker]
            
            # Returns features
            features_dict[f"{ticker}_return"] = returns
            features_dict[f"{ticker}_return_sq"] = returns ** 2
            features_dict[f"{ticker}_return_abs"] = np.abs(returns)
            
            # Moving averages
            if 'close' in df.columns:
                close = df['close']
                features_dict[f"{ticker}_sma_5"] = close.rolling(5).mean() / close - 1
                features_dict[f"{ticker}_sma_20"] = close.rolling(20).mean() / close - 1
                features_dict[f"{ticker}_ema_12"] = close.ewm(span=12).mean() / close - 1
            
            # Volatility
            features_dict[f"{ticker}_vol_5"] = returns.rolling(5).std()
            features_dict[f"{ticker}_vol_20"] = returns.rolling(20).std()
            
            # Momentum
            features_dict[f"{ticker}_mom_5"] = returns.rolling(5).sum()
            features_dict[f"{ticker}_mom_20"] = returns.rolling(20).sum()
            
            # Range features
            if all(col in df.columns for col in ['high', 'low', 'close']):
                features_dict[f"{ticker}_hl_range"] = (df['high'] - df['low']) / df['close']
            
            # Volume features
            if 'volume' in df.columns:
                volume = df['volume']
                features_dict[f"{ticker}_volume_change"] = volume.pct_change()
                features_dict[f"{ticker}_volume_ma_ratio"] = volume / volume.rolling(20).mean()
        
        return pd.DataFrame(features_dict)
    
    def _align_data(
        self,
        returns_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align data."""
        common_idx = returns_df.index.intersection(features_df.index)
        returns_df = returns_df.loc[common_idx]
        features_df = features_df.loc[common_idx]
        
        valid_idx = returns_df.dropna().index.intersection(features_df.dropna().index)
        return returns_df.loc[valid_idx], features_df.loc[valid_idx]
    
    def _split_train_test(
        self,
        returns_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> None:
        """Split data."""
        test_ratio = self.config.get('data', {}).get('test_ratio', 0.2)
        n = len(returns_df)
        split_idx = int(n * (1 - test_ratio))
        
        self.bundle.returns_train = returns_df.iloc[:split_idx]
        self.bundle.returns_test = returns_df.iloc[split_idx:]
        self.bundle.features_train = features_df.iloc[:split_idx]
        self.bundle.features_test = features_df.iloc[split_idx:]
        self.bundle.X_train = features_df.iloc[:split_idx]
        self.bundle.X_test = features_df.iloc[split_idx:]
