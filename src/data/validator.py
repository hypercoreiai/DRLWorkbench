# Data validation — pre-flight checks (V3 — PROJECT_OUTLINE Section 2.1)
# Entry: validate_data(config) called before any modeling.

from typing import Optional, Union

import pandas as pd

from src.utils import DataValidationError


class DataValidator:
    """
    Pre-flight checks before pipeline execution.
    """

    def check_missing_data(
        self, df: pd.DataFrame, threshold: float = 0.05
    ) -> None:
        """
        Fail if fraction of NaNs exceeds threshold.

        Args:
            df: Input DataFrame.
            threshold: Max allowed fraction of missing values (default 5%).

        Raises:
            DataValidationError: If missing data exceeds threshold.
        """
        missing = df.isna().mean()
        if (missing > threshold).any():
            cols = missing[missing > threshold].index.tolist()
            raise DataValidationError(
                f"Columns exceed {threshold:.0%} missing: {cols}"
            )

    def check_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
    ) -> pd.DataFrame:
        """
        Flag or remove extremes (IQR or z-score).

        Args:
            df: Input DataFrame (numeric).
            method: 'iqr' or 'zscore'.

        Returns:
            DataFrame with outliers handled (e.g. clipped or dropped).
        """
        if method == "iqr":
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            return df.clip(lower=low, upper=high, axis=1)
        if method == "zscore":
            z = (df - df.mean()).abs() / df.std().replace(0, 1)
            return df.where(z <= 3, df.median())
        raise ValueError(f"Unknown method: {method}")

    def check_stationarity(self, series: pd.Series) -> bool:
        """
        ADF test for unit roots. Returns True if series appears stationary.

        Args:
            series: Time series to test.

        Returns:
            True if null of unit root is rejected (stationary).
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            return True
        series_clean = series.dropna()
        if len(series_clean) < 10:
            return True
        result = adfuller(series_clean, autolag="AIC")
        return result[1] < 0.05

    def check_data_leakage(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        X_test: Union[pd.DataFrame, pd.Series],
    ) -> None:
        """
        Ensure no temporal overlap between train and test.

        Args:
            X_train: Training features (must have index with dates if applicable).
            X_test: Test features.

        Raises:
            DataValidationError: If overlap detected.
        """
        if hasattr(X_train, "index") and hasattr(X_test, "index"):
            overlap = X_train.index.intersection(X_test.index)
            if len(overlap) > 0:
                raise DataValidationError(
                    f"Temporal overlap between train and test: {len(overlap)} rows"
                )

    def check_collinearity(
        self, df: pd.DataFrame, threshold: float = 0.9
    ) -> list:
        """
        Warn on multicollinearity (high correlation pairs).

        Args:
            df: Numeric DataFrame.
            threshold: Correlation threshold above which to flag.

        Returns:
            List of (col1, col2) pairs with correlation >= threshold.
        """
        corr = df.corr()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) >= threshold:
                    pairs.append((corr.columns[i], corr.columns[j]))
        return pairs

    def check_crypto_continuity(
        self, df: pd.DataFrame, freq: str = "1h"
    ) -> None:
        """
        Ensure 24/7 data continuity (no gaps > 1 period).
        
        Args:
            df: DataFrame with datetime index.
            freq: Expected frequency offset (e.g. '1h', '15min').
            
        Raises:
            DataValidationError: If gaps are found.
        """
        if not hasattr(df, "index") or not isinstance(df.index, pd.DatetimeIndex):
            return # Skip if not time series
            
        # Create full range
        start, end = df.index.min(), df.index.max()
        full_range = pd.date_range(start=start, end=end, freq=freq)
        
        # Check difference
        # Efficient way: ensure lengths match or check diffs
        if len(df) != len(full_range):
            missing_dates = full_range.difference(df.index)
            if not missing_dates.empty:
                 raise DataValidationError(
                    f"Crypto continuity breach: {len(missing_dates)} missing periods. "
                    f"First missing: {missing_dates[0]}"
                )
