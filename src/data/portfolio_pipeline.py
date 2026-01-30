"""
Portfolio Optimization Data Pipeline
Enhanced pipeline for multi-asset portfolio optimization with features and returns
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import APIError, DataValidationError
from src.data.pipeline import DataBundle


class PortfolioDataBundle(DataBundle):
    """
    Extended data bundle for portfolio optimization.
    Contains returns, features, and metadata for multiple assets.
    """
    
    def __init__(self):
        super().__init__()
        self.returns_train: Optional[pd.DataFrame] = None
        self.returns_test: Optional[pd.DataFrame] = None
        self.features_train: Optional[pd.DataFrame] = None
        self.features_test: Optional[pd.DataFrame] = None
        self.prices_train: Optional[pd.DataFrame] = None
        self.prices_test: Optional[pd.DataFrame] = None


class PortfolioPipeline:
    """
    Portfolio optimization data pipeline.
    
    Steps:
    1. Load multi-asset OHLCV data
    2. Calculate returns and technical indicators
    3. Align data across assets
    4. Split train/test
    5. Validate data quality
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bundle = PortfolioDataBundle()
        
    def run(self) -> PortfolioDataBundle:
        """
        Execute full pipeline.
        
        Returns:
            PortfolioDataBundle with returns, features, prices for train/test.
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
        
        # Calculate features (technical indicators)
        features_df = self._calculate_features(ohlcv_data)
        
        # Extract prices (close prices)
        prices_df = self._extract_prices(ohlcv_data)
        
        # Align data (same index)
        returns_df, features_df, prices_df = self._align_data(returns_df, features_df, prices_df)
        
        # Split train/test
        self._split_train_test(returns_df, features_df, prices_df)
        
        # Store metadata
        self.bundle.metadata['tickers'] = tickers
        self.bundle.metadata['n_assets'] = len(tickers)
        self.bundle.metadata['config'] = self.config
        
        return self.bundle
    
    def _load_tickers(self) -> List[str]:
        """Load ticker list from config or symbols file."""
        # Check if tickers specified in config
        tickers = self.config.get('data', {}).get('tickers', [])
        
        if tickers:
            return tickers
        
        # Load from symbols/portfolio file
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
        """
        Load OHLCV data for all tickers.
        
        Returns:
            Dict mapping ticker -> DataFrame with OHLCV columns.
        """
        from src.ohlcv.yfinance_ohlcv import YFinanceOHLCV
        
        yf = YFinanceOHLCV()
        period = self.config.get('data', {}).get('period', '2y')
        
        try:
            results = yf.get(tickers, period)
            
            # Filter out empty results
            valid_results = {k: v for k, v in results.items() if not v.empty}
            
            if not valid_results:
                self.bundle.error_log.append("No valid data returned from YFinance")
                
            return valid_results
            
        except Exception as e:
            self.bundle.error_log.append(f"Data download failed: {e}")
            return {}
    
    def _calculate_returns(self, ohlcv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate returns for all assets.
        
        Args:
            ohlcv_data: Dict of ticker -> OHLCV DataFrame
        
        Returns:
            DataFrame with columns = tickers, values = returns
        """
        returns_dict = {}
        
        for ticker, df in ohlcv_data.items():
            if 'close' in df.columns:
                returns = df['close'].pct_change()
                returns_dict[ticker] = returns
        
        # Combine into single DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Drop first row (NaN from pct_change)
        returns_df = returns_df.iloc[1:]
        
        return returns_df
    
    def _calculate_features(self, ohlcv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate technical indicators as features.
        
        Args:
            ohlcv_data: Dict of ticker -> OHLCV DataFrame
        
        Returns:
            DataFrame with multi-level columns (ticker, feature)
        """
        features_dict = {}
        
        for ticker, df in ohlcv_data.items():
            ticker_features = {}
            
            # Returns
            if 'close' in df.columns:
                ticker_features['returns'] = df['close'].pct_change()
                
                # Moving averages
                ticker_features['sma_10'] = df['close'].rolling(10).mean()
                ticker_features['sma_30'] = df['close'].rolling(30).mean()
                
                # Volatility
                ticker_features['volatility_10'] = df['close'].pct_change().rolling(10).std()
                ticker_features['volatility_30'] = df['close'].pct_change().rolling(30).std()
                
                # Momentum
                ticker_features['momentum_10'] = df['close'].pct_change(10)
                ticker_features['momentum_30'] = df['close'].pct_change(30)
            
            # Volume features
            if 'volume' in df.columns:
                ticker_features['volume'] = df['volume']
                ticker_features['volume_sma_10'] = df['volume'].rolling(10).mean()
            
            # High-Low spread
            if 'high' in df.columns and 'low' in df.columns:
                ticker_features['hl_spread'] = (df['high'] - df['low']) / df['low']
            
            # Convert to DataFrame
            ticker_df = pd.DataFrame(ticker_features, index=df.index)
            
            # Add ticker prefix to columns
            ticker_df.columns = [f"{ticker}_{col}" for col in ticker_df.columns]
            
            features_dict[ticker] = ticker_df
        
        # Concatenate all features
        if features_dict:
            features_df = pd.concat(features_dict.values(), axis=1)
        else:
            features_df = pd.DataFrame()
        
        return features_df
    
    def _extract_prices(self, ohlcv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extract close prices for all assets.
        
        Args:
            ohlcv_data: Dict of ticker -> OHLCV DataFrame
        
        Returns:
            DataFrame with columns = tickers, values = close prices
        """
        prices_dict = {}
        
        for ticker, df in ohlcv_data.items():
            if 'close' in df.columns:
                prices_dict[ticker] = df['close']
        
        prices_df = pd.DataFrame(prices_dict)
        
        return prices_df
    
    def _align_data(
        self,
        returns_df: pd.DataFrame,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Align all DataFrames to have the same index (inner join).
        
        Returns:
            Tuple of (returns, features, prices) all with same index.
        """
        # Get common index
        common_idx = returns_df.index.intersection(features_df.index).intersection(prices_df.index)
        
        # Reindex all
        returns_df = returns_df.loc[common_idx]
        features_df = features_df.loc[common_idx]
        prices_df = prices_df.loc[common_idx]
        
        # Drop any remaining NaN rows
        valid_idx = returns_df.dropna().index.intersection(features_df.dropna().index)
        
        returns_df = returns_df.loc[valid_idx]
        features_df = features_df.loc[valid_idx]
        prices_df = prices_df.loc[valid_idx]
        
        return returns_df, features_df, prices_df
    
    def _split_train_test(
        self,
        returns_df: pd.DataFrame,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ) -> None:
        """
        Split data into train/test sets (temporal split).
        
        Updates self.bundle with train/test data.
        """
        # Get split ratio
        test_ratio = self.config.get('data', {}).get('test_ratio', 0.2)
        
        # Calculate split index
        n = len(returns_df)
        split_idx = int(n * (1 - test_ratio))
        
        # Split
        self.bundle.returns_train = returns_df.iloc[:split_idx]
        self.bundle.returns_test = returns_df.iloc[split_idx:]
        
        self.bundle.features_train = features_df.iloc[:split_idx]
        self.bundle.features_test = features_df.iloc[split_idx:]
        
        self.bundle.prices_train = prices_df.iloc[:split_idx]
        self.bundle.prices_test = prices_df.iloc[split_idx:]
        
        # Store in legacy X_train/X_test for compatibility
        self.bundle.X_train = features_df.iloc[:split_idx]
        self.bundle.X_test = features_df.iloc[split_idx:]


def load_index_data(tickers: List[str], period: str = '2y') -> pd.DataFrame:
    """
    Load benchmark index data (e.g., SPY, BTC for crypto).
    
    Args:
        tickers: List of index tickers.
        period: Time period to load.
    
    Returns:
        DataFrame with index returns.
    """
    from src.ohlcv.yfinance_ohlcv import YFinanceOHLCV
    
    yf = YFinanceOHLCV()
    
    try:
        results = yf.get(tickers, period)
        
        # Calculate returns
        returns_dict = {}
        for ticker, df in results.items():
            if not df.empty and 'close' in df.columns:
                returns = df['close'].pct_change()
                returns_dict[ticker] = returns
        
        return pd.DataFrame(returns_dict)
        
    except Exception as e:
        print(f"Failed to load index data: {e}")
        return pd.DataFrame()
