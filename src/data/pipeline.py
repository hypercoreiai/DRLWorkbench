# Unified data pipeline with error recovery (V3 — PROJECT_OUTLINE Section 2.3)
# DataPipeline(config): download → clean → indicators → validate → filter → scale → sequences

from typing import Any, Dict, Optional

from src.utils import APIError, DataValidationError


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

    def run(self) -> DataBundle:
        """
        Execute full pipeline. Populates self.bundle.
        """
        # Stub: actual implementation will wire to ohlcv, indicators, normalize.
        self.bundle.metadata["config"] = self.config
        
        tickers = self.config.get("data", {}).get("tickers", [])
        if not tickers:
            print("No tickers configured in data.tickers")
            return self.bundle

        # Simple implementation: use YFinance for now as it supports most tickers
        from src.ohlcv.yfinance_ohlcv import YFinanceOHLCV
        import pandas as pd
        
        yf = YFinanceOHLCV()
        period = self.config.get("data", {}).get("period", "2y")
        
        # Download all
        try:
            results = yf.get(tickers, period)
        except Exception as e:
            self.bundle.error_log.append(f"Data download failed: {e}")
            return self.bundle

        # Combine or pick primary. For this stage, let's concat or picking the first valid one if only one expected?
        # The User's config shows [BTC-USD, ETH-USD].
        # If WalkForwardBacktester expects a single DF, we might need to handle this.
        # But for V3 Portfolio spec, we likely want a combined DF or dict.
        # However, `walker.py` expects a dataframe. 
        # Let's verify `walker.py` logic. It slices self.data.iloc.
        # If we pass a concatenated DF (MultiIndex?), slicing works.
        # If we pass a wide DF (close_BTC, close_ETH), slicing works.
        # Let's create a combined DataFrame for now, aligned on index.
        
        combined_df = pd.DataFrame()
        valid_dfs = []
        
        for ticker, df in results.items():
            if df.empty:
                continue
            # Rename columns to include ticker prefix if multiple
            if len(tickers) > 1:
                df = df.add_prefix(f"{ticker}_")
            valid_dfs.append(df)
            
        if valid_dfs:
            # Join on index (outer or inner? Inner for aligned data)
            combined_df = pd.concat(valid_dfs, axis=1, join="inner")
            self.bundle.X_train = combined_df # Store into X_train which run_pipeline uses
            self.bundle.metadata["tickers"] = list(results.keys())
        else:
             self.bundle.error_log.append("No valid data found for tickers")
             
        return self.bundle
