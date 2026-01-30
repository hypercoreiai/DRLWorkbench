# Unit tests for OHLCV daily returns handler (data layer)

import numpy as np
import pandas as pd
import pytest

from src.data.ohlcv_returns import OHLCVDailyReturns, _normalize_interval, _is_daily_interval
from src.utils import DataValidationError


class TestOHLCVDailyReturns:
    def test_daily_candles_adds_daily_returns(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {"open": [100, 102, 101, 105, 104], "close": [101, 100, 103, 104, 106]},
            index=idx,
        )
        handler = OHLCVDailyReturns(interval="1d")
        out = handler.add_daily_returns(df)
        assert "daily_returns" in out.columns
        assert out["daily_returns"].iloc[0] == 0  # NaN filled with 0
        expected = df["close"].pct_change().fillna(0)
        pd.testing.assert_series_equal(out["daily_returns"], expected, check_names=False)

    def test_daily_candles_close_column_Close(self):
        idx = pd.date_range("2020-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {"Open": [10, 11, 12, 13], "Close": [10.5, 11.2, 11.8, 12.5]},
            index=idx,
        )
        handler = OHLCVDailyReturns(interval="1d")
        out = handler.add_daily_returns(df)
        assert "daily_returns" in out.columns
        expected = df["Close"].pct_change().fillna(0)
        pd.testing.assert_series_equal(out["daily_returns"], expected, check_names=False)

    def test_subdaily_1h_adds_daily_returns(self):
        # 3 days of hourly data: 3 * 24 = 72 bars
        idx = pd.date_range("2020-01-01 00:00:00", periods=72, freq="1h")
        close = 100.0 + np.cumsum(np.random.randn(72) * 0.1)
        df = pd.DataFrame({"close": close}, index=idx)
        handler = OHLCVDailyReturns(interval="1h")
        out = handler.add_daily_returns(df)
        assert "daily_returns" in out.columns
        # First day has no previous close -> filled with 0
        assert out["daily_returns"].iloc[0] == 0
        # All bars on same day should have same daily_returns
        day1_bars = out.loc["2020-01-01"]
        day2_bars = out.loc["2020-01-02"]
        day3_bars = out.loc["2020-01-03"]
        assert (day1_bars["daily_returns"] == 0).all()
        assert day2_bars["daily_returns"].nunique() == 1
        assert day3_bars["daily_returns"].nunique() == 1
        # Daily return day2 = (last close day2 / last close day1) - 1
        last_close_d1 = df.loc["2020-01-01", "close"].iloc[-1]
        last_close_d2 = df.loc["2020-01-02", "close"].iloc[-1]
        expected_d2 = (last_close_d2 / last_close_d1) - 1
        assert abs(day2_bars["daily_returns"].iloc[0] - expected_d2) < 1e-10

    def test_subdaily_15m_interval(self):
        idx = pd.date_range("2020-01-01 00:00:00", periods=4 * 24, freq="15min")  # 1 day of 15m bars
        close = 100.0 + np.arange(len(idx)) * 0.01
        df = pd.DataFrame({"close": close}, index=idx)
        handler = OHLCVDailyReturns(interval="15m")
        out = handler.add_daily_returns(df)
        assert "daily_returns" in out.columns
        # Only one day -> first day has daily return filled with 0 for all bars
        assert (out["daily_returns"] == 0).all()

    def test_empty_df_raises(self):
        handler = OHLCVDailyReturns(interval="1d")
        df = pd.DataFrame({"close": []})
        df.index = pd.DatetimeIndex([])
        with pytest.raises(DataValidationError):
            handler.add_daily_returns(df)

    def test_no_close_column_raises(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame({"open": [1, 2, 3], "high": [2, 3, 4]}, index=idx)
        handler = OHLCVDailyReturns(interval="1d")
        with pytest.raises(DataValidationError):
            handler.add_daily_returns(df)

    def test_no_datetime_index_raises(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        handler = OHLCVDailyReturns(interval="1d")
        with pytest.raises(DataValidationError):
            handler.add_daily_returns(df)

    def test_unsupported_interval_raises(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame({"close": [100, 101, 102]}, index=idx)
        handler = OHLCVDailyReturns(interval="1w")
        with pytest.raises(DataValidationError):
            handler.add_daily_returns(df)

    def test_inplace(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame({"close": [100, 101, 102]}, index=idx)
        handler = OHLCVDailyReturns(interval="1d")
        out = handler.add_daily_returns(df, inplace=True)
        assert out is df
        assert "daily_returns" in df.columns

    def test_interval_aliases_1m_15m_4h(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=idx)
        for interval in ("1d", "1D", "1day", "daily"):
            handler = OHLCVDailyReturns(interval=interval)
            out = handler.add_daily_returns(df)
            assert "daily_returns" in out.columns
            pd.testing.assert_series_equal(out["daily_returns"], df["close"].pct_change().fillna(0), check_names=False)


class TestAddDailyReturnsIfApplicable:
    """Tests for add_daily_returns_if_applicable (graceful, single-column, date-as-column)."""

    def test_single_column_date_as_column_adds_daily_returns(self):
        # Indicator-style: columns [date, value], no DatetimeIndex
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "value": [100.0, 101.0, 102.0, 101.5, 103.0],
        })
        handler = OHLCVDailyReturns(interval="1d")
        out = handler.add_daily_returns_if_applicable(df)
        assert "daily_returns" in out.columns
        assert "date" in out.columns
        assert "value" in out.columns
        assert out["daily_returns"].iloc[0] == 0  # NaN filled with 0
        expected = df["value"].pct_change().fillna(0)
        pd.testing.assert_series_equal(out["daily_returns"], expected, check_names=False)

    def test_single_column_no_date_column_returns_unchanged(self):
        df = pd.DataFrame({"value": [100, 101, 102]})  # no DatetimeIndex, no date column
        handler = OHLCVDailyReturns(interval="1d")
        out = handler.add_daily_returns_if_applicable(df)
        assert "daily_returns" not in out.columns
        pd.testing.assert_frame_equal(out, df)

    def test_empty_df_returns_unchanged(self):
        df = pd.DataFrame({"close": []})
        df.index = pd.DatetimeIndex([])
        handler = OHLCVDailyReturns(interval="1d")
        out = handler.add_daily_returns_if_applicable(df)
        assert out is not None
        assert out.empty

    def test_no_numeric_column_returns_unchanged(self):
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3, freq="D"), "label": ["a", "b", "c"]})
        handler = OHLCVDailyReturns(interval="1d")
        out = handler.add_daily_returns_if_applicable(df)
        assert "daily_returns" not in out.columns
        pd.testing.assert_frame_equal(out, df)


class TestNormalizeInterval:
    def test_1d_aliases(self):
        assert _normalize_interval("1d") == "1d"
        assert _normalize_interval("1D") == "1d"
        assert _normalize_interval("daily") == "1d"

    def test_subdaily(self):
        assert _normalize_interval("1m") == "1m"
        assert _normalize_interval("15m") == "15m"
        assert _normalize_interval("1h") == "1h"
        assert _normalize_interval("4h") == "4h"

    def test_is_daily(self):
        assert _is_daily_interval("1d") is True
        assert _is_daily_interval("1D") is True
        assert _is_daily_interval("1h") is False
        assert _is_daily_interval("15m") is False
