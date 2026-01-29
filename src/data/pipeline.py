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
        return self.bundle
