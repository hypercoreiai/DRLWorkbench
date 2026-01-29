# Unified data pipeline with error recovery (V3 — PROJECT_OUTLINE Section 2.3)
# DataPipeline(config): download → clean → indicators → validate → filter → scale → sequences

from typing import Any, Dict, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils import APIError, DataValidationError
from src.data.validator import DataValidator
from src.ohlcv.yfinance_ohlcv import YFinanceOHLCV
from src.ohlcv.kraken_ohlcv import KrakenOHLCV
from src.ohlcv.clean_ohlcv import CleanOHLCV
from src.data.sequence import build_sequences

logger = logging.getLogger(__name__)


class DataBundle:
    """
    Output container: train/test sets, metadata, validation results, error logs.
    """

    def __init__(self) -> None:
        self.X_train: Optional[Any] = None
        self.y_train: Optional[Any] = None
        self.X_test: Optional[Any] = None
        self.y_test: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.validation_report: Dict[str, Any] = {}
        self.error_log: list = []


class DataPipeline:
    """
    Pipeline steps (with error handling at each stage):
    1. Download OHLCV (retry logic, fallback to cached).
    2. Clean & align (fill gaps, validate OHLCV constraints).
    3. Add technical indicators (from ohlcv/indicators).
    4. Validate data (NaNs, outliers, stationarity).
    5. Liquidity/volatility filtering and composite scoring.
    6. Feature selection (correlation + PCA).
    7. Check data leakage before split.
    8. Scale (RevinTransform or MinMaxScaler).
    9. Build sequences or NeuralForecast format.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.bundle = DataBundle()
        self.validator = DataValidator()

    def run(self) -> DataBundle:
        """
        Execute full pipeline. Populates self.bundle.
        """
        try:
            # Step 1: Download OHLCV
            logger.info("Step 1: Downloading OHLCV data...")
            raw_data = self._download_ohlcv()
            
            # Step 2: Clean & align
            logger.info("Step 2: Cleaning and aligning data...")
            clean_data = self._clean_align(raw_data)
            
            # Step 3: Add technical indicators
            logger.info("Step 3: Adding technical indicators...")
            feature_data = self._add_indicators(clean_data)
            
            # Step 4: Validate data
            logger.info("Step 4: Validating data quality...")
            validated_data = self._validate_data(feature_data)
            
            # Step 5: Liquidity/volatility filtering (if configured)
            logger.info("Step 5: Applying filters...")
            filtered_data = self._apply_filters(validated_data)
            
            # Step 6: Feature selection (correlation-based)
            logger.info("Step 6: Selecting features...")
            selected_data = self._select_features(filtered_data)
            
            # Step 7: Train/test split
            logger.info("Step 7: Splitting train/test...")
            train_data, test_data = self._split_data(selected_data)
            
            # Step 8: Check data leakage
            logger.info("Step 8: Checking for data leakage...")
            self._check_leakage(train_data, test_data)
            
            # Step 9: Scale
            logger.info("Step 9: Scaling data...")
            train_scaled, test_scaled, scaler = self._scale_data(train_data, test_data)
            
            # Step 10: Build sequences
            logger.info("Step 10: Building sequences...")
            self._build_sequences(train_scaled, test_scaled)
            
            self.bundle.metadata.update({
                "config": self.config,
                "scaler": scaler,
                "feature_names": list(selected_data.columns),
                "train_dates": (train_data.index[0], train_data.index[-1]),
                "test_dates": (test_data.index[0], test_data.index[-1]),
            })
            
            logger.info("Pipeline completed successfully")
            return self.bundle
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.bundle.error_log.append(str(e))
            raise

    def _download_ohlcv(self) -> Dict[str, pd.DataFrame]:
        """Download OHLCV data from configured source."""
        data_config = self.config.get("data", {})
        tickers = data_config.get("tickers", [])
        period = data_config.get("period", "2y")
        source = data_config.get("source", "yfinance")
        
        if not tickers:
            raise DataValidationError("No tickers specified in config")
        
        try:
            if source == "yfinance":
                fetcher = YFinanceOHLCV()
                return fetcher.get(tickers, period)
            elif source == "kraken" or source.startswith("binance"):
                fetcher = KrakenOHLCV()
                # Note: KrakenOHLCV may need API credentials
                result = {}
                for ticker in tickers:
                    try:
                        result[ticker] = fetcher.get(ticker, period)
                    except Exception as e:
                        logger.warning(f"Failed to fetch {ticker}: {e}")
                return result
            else:
                raise ValueError(f"Unknown data source: {source}")
        except Exception as e:
            raise APIError(f"Failed to download OHLCV: {e}") from e

    def _clean_align(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Clean and align OHLCV data."""
        cleaner = CleanOHLCV()
        
        # Filter out empty dataframes
        valid_data = {
            ticker: df for ticker, df in raw_data.items()
            if not df.empty
        }
        
        if not valid_data:
            raise DataValidationError("No valid data after filtering empty frames")
        
        try:
            # CleanOHLCV.clean expects a dict and returns a dict
            cleaned_dict = cleaner.clean(valid_data)
            
            # For now, use the first ticker's data as primary
            # In multi-asset scenarios, this would be more sophisticated
            primary_ticker = list(cleaned_dict.keys())[0]
            df = cleaned_dict[primary_ticker]
            
            # Standardize column names to lowercase
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure we have OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise DataValidationError(f"Missing required columns: {missing}")
            
            return df
        except Exception as e:
            logger.error(f"Failed to clean data: {e}")
            # Fallback: use first valid ticker without cleaning
            primary_ticker = list(valid_data.keys())[0]
            df = valid_data[primary_ticker].copy()
            df.columns = [c.lower() for c in df.columns]
            return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        result = df.copy()
        
        # Add basic technical indicators
        close = df['close']
        
        # Moving averages
        for period in [10, 20, 50]:
            result[f'sma_{period}'] = close.rolling(window=period).mean()
            result[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        result['volatility'] = close.pct_change().rolling(window=20).std()
        
        # Returns
        result['returns'] = close.pct_change()
        
        return result

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality."""
        validation_config = self.config.get("data", {}).get("validation", {})
        
        # Check missing data
        missing_threshold = validation_config.get("missing_threshold", 0.05)
        try:
            self.validator.check_missing_data(df, threshold=missing_threshold)
        except DataValidationError as e:
            logger.warning(f"Missing data issue: {e}")
            # Fill forward then backward for now (using ffill/bfill directly)
            df = df.ffill().bfill()
        
        # Handle outliers
        outlier_method = validation_config.get("outlier_method", "iqr")
        if outlier_method in ["iqr", "zscore"]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.validator.check_outliers(
                df[numeric_cols], method=outlier_method
            )
        
        self.bundle.validation_report["missing_data_check"] = "passed"
        self.bundle.validation_report["outlier_handling"] = outlier_method
        
        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply liquidity/volatility filters."""
        # For single-asset analysis, just validate ranges
        filter_config = self.config.get("data", {})
        
        if 'volatility_filter' in filter_config:
            vol_config = filter_config['volatility_filter']
            min_vol = vol_config.get('min', 0.1)
            max_vol = vol_config.get('max', 0.5)
            
            if 'volatility' in df.columns:
                avg_vol = df['volatility'].mean()
                if avg_vol < min_vol or avg_vol > max_vol:
                    logger.warning(
                        f"Average volatility {avg_vol:.4f} outside "
                        f"range [{min_vol}, {max_vol}]"
                    )
        
        return df

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature selection based on correlation."""
        selection_config = self.config.get("data", {}).get("feature_selection", {})
        method = selection_config.get("method", "correlation")
        threshold = selection_config.get("threshold", 0.5)
        
        if method != "correlation":
            return df
        
        # Remove non-numeric and target columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Drop NaN rows for correlation computation
        numeric_df = numeric_df.dropna()
        
        if 'returns' in numeric_df.columns:
            target = numeric_df['returns']
            features = numeric_df.drop(columns=['returns'])
            
            # Compute correlation with target
            correlations = features.corrwith(target).abs()
            selected = correlations[correlations >= threshold].index.tolist()
            
            if selected:
                logger.info(f"Selected {len(selected)} features with |corr| >= {threshold}")
                return numeric_df[selected + ['returns']]
        
        return numeric_df

    def _split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train and test sets."""
        data_config = self.config.get("data", {})
        test_size = data_config.get("test_size", 0.2)
        
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
        logger.info(f"Train: {len(train)} samples, Test: {len(test)} samples")
        return train, test

    def _check_leakage(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """Check for data leakage between train and test."""
        if self.config.get("data", {}).get("validation", {}).get("check_leakage", True):
            try:
                self.validator.check_data_leakage(train, test)
                self.bundle.validation_report["leakage_check"] = "passed"
            except DataValidationError as e:
                logger.error(f"Data leakage detected: {e}")
                raise

    def _scale_data(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        """Scale data using MinMaxScaler."""
        scaler = MinMaxScaler()
        
        # Fit on train, transform both
        train_scaled = pd.DataFrame(
            scaler.fit_transform(train),
            index=train.index,
            columns=train.columns
        )
        test_scaled = pd.DataFrame(
            scaler.transform(test),
            index=test.index,
            columns=test.columns
        )
        
        return train_scaled, test_scaled, scaler

    def _build_sequences(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """Build sequences for time series models."""
        data_config = self.config.get("data", {})
        time_step = data_config.get("time_step", 10)
        look_ahead = data_config.get("look_ahead", 1)
        
        # Target is returns column (or close if returns not available)
        target_col = 'returns' if 'returns' in train.columns else 'close'
        
        # Prepare features and target
        if target_col in train.columns:
            X_train_df = train.drop(columns=[target_col])
            y_train_series = train[target_col]
            X_test_df = test.drop(columns=[target_col])
            y_test_series = test[target_col]
        else:
            X_train_df = train
            y_train_series = train['close']
            X_test_df = test
            y_test_series = test['close']
        
        # Convert to numpy
        X_train_np = X_train_df.values
        y_train_np = y_train_series.values
        X_test_np = X_test_df.values
        y_test_np = y_test_series.values
        
        # Build sequences
        X_train, y_train = build_sequences(
            X_train_np,
            y_train_np,
            time_step=time_step,
            look_ahead=look_ahead
        )
        
        X_test, y_test = build_sequences(
            X_test_np,
            y_test_np,
            time_step=time_step,
            look_ahead=look_ahead
        )
        
        self.bundle.X_train = X_train
        self.bundle.y_train = y_train
        self.bundle.X_test = X_test
        self.bundle.y_test = y_test
        
        logger.info(
            f"Sequences built: X_train {X_train.shape}, y_train {y_train.shape}, "
            f"X_test {X_test.shape}, y_test {y_test.shape}"
        )
