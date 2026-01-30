# Data handling: OHLCV with daily returns (V3/V4 — Notes design)
# Accepts OHLCV at subdaily or daily intervals and adds a daily_returns column.

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from src.utils import DataValidationError


# Supported bar intervals: 1m, 15m, 1h, 4h, 1d (and common aliases)
SUPPORTED_INTERVALS = frozenset({
    "1m", "1min", "1minute",
    "15m", "15min", "15minute",
    "1h", "1hr", "1hour", "60m", "60min",
    "4h", "4hr", "4hour", "240m", "240min",
    "1d", "1D", "1day", "daily", "1440m",
})
DAILY_INTERVALS = frozenset({"1d", "1D", "1day", "daily", "1440m"})


def _normalize_interval(interval: str) -> str:
    """Return canonical interval string (lowercase, standard form)."""
    s = str(interval).strip().lower()
    if s in ("1d", "1day", "daily"):
        return "1d"
    if s in ("1m", "1min", "1minute"):
        return "1m"
    if s in ("15m", "15min", "15minute"):
        return "15m"
    if s in ("1h", "1hr", "1hour", "60m", "60min"):
        return "1h"
    if s in ("4h", "4hr", "4hour", "240m", "240min"):
        return "4h"
    if s == "1440m":
        return "1d"
    return s


def _is_daily_interval(interval: str) -> bool:
    return _normalize_interval(interval) == "1d"


def _infer_interval_from_index(index: pd.DatetimeIndex) -> Optional[str]:
    """Infer bar interval from datetime index (pandas inferred freq)."""
    if index is None or len(index) < 2:
        return None
    freq = getattr(index, "inferred_freq", None) or pd.infer_freq(index)
    if freq is None:
        return None
    # Map common pandas freqs to our canonical names
    m = {
        "D": "1d", "B": "1d", "1D": "1d",
        "1min": "1m", "1T": "1m",
        "15min": "15m", "15T": "15m",
        "1h": "1h", "1H": "1h", "60min": "1h", "60T": "1h",
        "4h": "4h", "4H": "4h", "240min": "4h", "240T": "4h",
    }
    return m.get(freq, freq)


class OHLCVDailyReturns:
    """
    Data handler that accepts OHLCV data and adds a daily returns column.

    Works with subdaily intervals (1m, 15m, 1h, 4h) and daily candles (1d).
    For subdaily data, daily return is (last close of day / last close of
    previous day) - 1, and that value is attached to every bar of the day.
    For daily data, daily return is close.pct_change().

    Design (Notes V3/V4):
    - Accepts DataFrame with datetime index and a close column (e.g. close or Close).
    - Interval can be provided or inferred from the index.
    - Output column name: daily_returns.
    """

    # Column names accepted for close price (first match wins)
    CLOSE_ALIASES = ("close", "Close")

    def __init__(
        self,
        interval: Optional[str] = None,
        close_column: Optional[str] = None,
    ) -> None:
        """
        Initialize the handler.

        Parameters
        ----------
        interval : str, optional
            Bar interval: "1m", "15m", "1h", "4h", "1d" (or aliases).
            If None, interval is inferred from the DataFrame index when possible.
        close_column : str, optional
            Name of the close price column. If None, first of ("close", "Close") is used.
        """
        self.interval = interval
        self.close_column = close_column

    def add_daily_returns(
        self,
        df: pd.DataFrame,
        interval: Optional[str] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Add a daily_returns column to OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with datetime index and a close column.
        interval : str, optional
            Override bar interval for this call ("1m", "15m", "1h", "4h", "1d").
            Uses instance interval or infers from index if None.
        inplace : bool, default False
            If True, modify df in place and return it.

        Returns
        -------
        pd.DataFrame
            DataFrame with added column "daily_returns". First calendar day
            (or first row for daily data) will have NaN (no prior close).

        Raises
        ------
        DataValidationError
            If required columns/index are missing or interval is unsupported.
        """
        if df is None or df.empty:
            raise DataValidationError("OHLCV DataFrame is empty or None")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError(
                "OHLCV DataFrame must have a DatetimeIndex"
            )

        close_col = self._resolve_close_column(df)
        if close_col is None:
            raise DataValidationError(
                f"OHLCV must have a close column; found columns: {list(df.columns)}"
            )

        resolved_interval = interval or self.interval
        if resolved_interval is None:
            resolved_interval = _infer_interval_from_index(df.index)
        if resolved_interval is None:
            raise DataValidationError(
                "Interval could not be inferred from index; pass interval explicitly"
            )

        normalized = _normalize_interval(resolved_interval)
        if normalized not in {"1m", "15m", "1h", "4h", "1d"}:
            raise DataValidationError(
                f"Unsupported interval {resolved_interval!r}; "
                f"supported: 1m, 15m, 1h, 4h, 1d"
            )

        out = df if inplace else df.copy()

        if _is_daily_interval(resolved_interval):
            out["daily_returns"] = out[close_col].pct_change()
        else:
            # Subdaily: daily return = (last close today / last close yesterday) - 1
            # Use calendar day (1D) so crypto 24/7 is consistent
            daily_close = out[close_col].resample("1D").last()
            daily_returns = daily_close.pct_change()
            # Map daily return back to each bar: same value for all bars in that day
            dates = out.index.normalize()
            out["daily_returns"] = daily_returns.reindex(dates).values

        out["daily_returns"] = out["daily_returns"].fillna(0)
        return out

    def _resolve_close_column(self, df: pd.DataFrame) -> Optional[str]:
        """Return close column name if present."""
        for alias in self.CLOSE_ALIASES:
            if alias in df.columns:
                return alias
        return None

    def _resolve_value_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Return column to use for daily returns: close/Close first, else the
        single numeric column (e.g. 'value' for indicator data).
        """
        close_col = self._resolve_close_column(df)
        if close_col is not None:
            return close_col
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric) == 1:
            return numeric[0]
        return None

    def add_daily_returns_if_applicable(
        self,
        df: pd.DataFrame,
        interval: Optional[str] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Add daily_returns when possible; return df unchanged otherwise.
        Never raises. Handles single-column (e.g. indicator) data and
        date-as-column (no DatetimeIndex) DataFrames.

        - If index is not DatetimeIndex but a 'date' column exists, uses it
          as index for computation then resets index so date stays a column.
        - If there is no 'close' but exactly one numeric column (e.g. 'value'),
          uses that column as the series for daily returns (assumes daily if
          interval cannot be inferred).
        """
        if df is None or df.empty:
            return df if inplace else (df.copy() if df is not None else pd.DataFrame())

        out = df if inplace else df.copy()
        value_col = self._resolve_value_column(out)
        if value_col is None:
            return out

        index_is_datetime = isinstance(out.index, pd.DatetimeIndex)
        date_column = "date" if "date" in out.columns and not index_is_datetime else None
        reset_index_after = False

        if not index_is_datetime and date_column:
            try:
                out = out.set_index(pd.to_datetime(out[date_column], errors="coerce"))
                if date_column in out.columns:
                    out = out.drop(columns=[date_column])
                out = out[out.index.notna()]
                out = out.dropna(axis=0, how="all", subset=[value_col])
                if out.empty or len(out) < 2:
                    return df if inplace else df.copy()
                reset_index_after = True
            except Exception:
                return df if inplace else df.copy()
        elif not index_is_datetime:
            return df if inplace else df.copy()

        resolved_interval = interval or self.interval
        if resolved_interval is None:
            resolved_interval = _infer_interval_from_index(out.index)
        if resolved_interval is None:
            resolved_interval = "1d"

        normalized = _normalize_interval(resolved_interval)
        if normalized not in {"1m", "15m", "1h", "4h", "1d"}:
            if reset_index_after:
                out = out.reset_index()
            return df if inplace else df.copy()

        try:
            if _is_daily_interval(resolved_interval):
                out["daily_returns"] = out[value_col].pct_change()
            else:
                daily_close = out[value_col].resample("1D").last()
                daily_returns = daily_close.pct_change()
                dates = out.index.normalize()
                out["daily_returns"] = daily_returns.reindex(dates).values
            out["daily_returns"] = out["daily_returns"].fillna(0)
        except Exception:
            if reset_index_after:
                out = out.reset_index()
            return df if inplace else df.copy()

        if reset_index_after:
            out = out.reset_index()
        return out
