"""
PatchTST data preparation pipeline: download, clean, feature engineering,
target creation, scaling, and sequence generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.portfolio_pipeline import load_index_data
from src.utils import DataValidationError


@dataclass
class PatchTSTDataBundle:
    features: pd.DataFrame
    target_price: pd.Series
    target_returns: pd.Series
    benchmarks: pd.DataFrame
    metadata: Dict[str, Any]


class StandardScaler:
    """Simple standard scaler for DataFrame/Series."""

    def __init__(self) -> None:
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None

    def fit(self, data: pd.DataFrame | pd.Series) -> None:
        if isinstance(data, pd.Series):
            self.mean_ = pd.Series({"value": data.mean()})
            self.std_ = pd.Series({"value": data.std()})
        else:
            self.mean_ = data.mean(axis=0)
            self.std_ = data.std(axis=0)

    def transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fit.")
        if isinstance(data, pd.Series):
            return (data - self.mean_["value"]) / (self.std_["value"] + 1e-8)
        return (data - self.mean_) / (self.std_ + 1e-8)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fit.")
        mean = self.mean_.values[0]
        std = self.std_.values[0]
        return data * (std + 1e-8) + mean


class PatchTSTDataPipeline:
    """Data pipeline for PatchTST forecasting."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self) -> PatchTSTDataBundle:
        tickers = self._load_tickers()
        if not tickers:
            raise DataValidationError("No tickers found for PatchTST pipeline")

        primary_ticker = self.config.get("data", {}).get("primary_ticker", tickers[0])
        period = self.config.get("data", {}).get("period", "2y")

        ohlcv_data = self._load_ohlcv_data(tickers, period)
        if primary_ticker not in ohlcv_data:
            raise DataValidationError(f"Primary ticker {primary_ticker} not found in OHLCV data")

        ohlcv_data = {k: self._clean_ohlcv(v) for k, v in ohlcv_data.items() if not v.empty}
        features = self._build_features(ohlcv_data)

        primary_df = ohlcv_data[primary_ticker]
        target_price = primary_df["close"].copy()
        target_returns = primary_df["close"].pct_change()

        # Align
        common_index = features.index.intersection(target_price.index)
        features = features.loc[common_index]
        target_price = target_price.loc[common_index]
        target_returns = target_returns.loc[common_index]

        # Drop NaNs
        valid_idx = features.dropna().index.intersection(target_price.dropna().index)
        features = features.loc[valid_idx]
        target_price = target_price.loc[valid_idx]
        target_returns = target_returns.loc[valid_idx]

        # Load benchmarks
        benchmarks = self._load_benchmarks(period)
        if not benchmarks.empty:
            benchmarks = benchmarks.loc[benchmarks.index.intersection(features.index)]

        metadata = {
            "tickers": tickers,
            "primary_ticker": primary_ticker,
            "period": period,
            "n_features": features.shape[1],
        }

        return PatchTSTDataBundle(
            features=features,
            target_price=target_price,
            target_returns=target_returns,
            benchmarks=benchmarks,
            metadata=metadata,
        )

    def _load_tickers(self) -> List[str]:
        tickers = self.config.get("data", {}).get("tickers", [])
        if tickers:
            return tickers
        symbols_path = self.config.get("data", {}).get("symbols_file", "src/symbols/portfolio")
        path = Path(symbols_path)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_ohlcv_data(self, tickers: List[str], period: str) -> Dict[str, pd.DataFrame]:
        from src.ohlcv.yfinance_ohlcv import YFinanceOHLCV

        yf = YFinanceOHLCV()
        data = yf.get(tickers, period)
        return {k: v for k, v in data.items() if v is not None and not v.empty}

    def _load_benchmarks(self, period: str) -> pd.DataFrame:
        benchmarks = self.config.get("data", {}).get("benchmarks")
        if not benchmarks:
            benchmarks = ["BTC-USD"]
        return load_index_data(benchmarks, period=period)

    def _clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = df.ffill().bfill()
        # Outlier clipping (IQR) on numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            df[col] = df[col].clip(lower, upper)
        return df

    def _build_features(self, ohlcv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        features = []
        for ticker, df in ohlcv_data.items():
            fe = FeatureEngineer(df)
            ticker_features = fe.engineer_features()
            ticker_features.columns = [f"{ticker}_{c}" for c in ticker_features.columns]
            features.append(ticker_features)
        if not features:
            return pd.DataFrame()
        return pd.concat(features, axis=1)


class FeatureEngineer:
    """Create technical indicators and features."""

    def __init__(self, ohlcv_df: pd.DataFrame):
        self.ohlcv = ohlcv_df.copy()
        self.features = pd.DataFrame(index=ohlcv_df.index)

    def engineer_features(self) -> pd.DataFrame:
        self.add_momentum_indicators()
        self.add_trend_indicators()
        self.add_volatility_indicators()
        self.add_volume_indicators()
        self.add_price_action_features()
        self.add_statistical_features()
        return self.features

    def add_momentum_indicators(self) -> None:
        close = self.ohlcv["close"]
        self.features["rsi_14"] = _rsi(close, 14)
        self.features["rsi_21"] = _rsi(close, 21)
        self.features["rsi_28"] = _rsi(close, 28)
        stoch_k, stoch_d = _stoch_rsi(close)
        self.features["stoch_rsi_k"] = stoch_k
        self.features["stoch_rsi_d"] = stoch_d
        macd, macd_signal, macd_hist = _macd(close)
        self.features["macd"] = macd
        self.features["macd_signal"] = macd_signal
        self.features["macd_hist"] = macd_hist
        self.features["roc_12"] = close.pct_change(12)
        self.features["roc_24"] = close.pct_change(24)
        self.features["mom_10"] = close - close.shift(10)
        self.features["mom_20"] = close - close.shift(20)

    def add_trend_indicators(self) -> None:
        close = self.ohlcv["close"]
        self.features["sma_20"] = close.rolling(20).mean()
        self.features["sma_50"] = close.rolling(50).mean()
        self.features["sma_100"] = close.rolling(100).mean()
        self.features["sma_200"] = close.rolling(200).mean()
        self.features["ema_12"] = close.ewm(span=12, adjust=False).mean()
        self.features["ema_26"] = close.ewm(span=26, adjust=False).mean()
        self.features["adx_14"] = _adx(self.ohlcv, 14)
        self.features["linreg_slope_20"] = _rolling_slope(close, 20)
        self.features["linreg_slope_50"] = _rolling_slope(close, 50)
        self.features["tema_14"] = _tema(close, 14)

    def add_volatility_indicators(self) -> None:
        close = self.ohlcv["close"]
        ma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        self.features["bb_upper"] = upper
        self.features["bb_lower"] = lower
        self.features["bb_percent"] = (close - lower) / (upper - lower + 1e-8)
        self.features["atr_14"] = _atr(self.ohlcv, 14)
        self.features["std_20"] = std
        self.features["cv_20"] = std / (ma + 1e-8)

    def add_volume_indicators(self) -> None:
        if "volume" not in self.ohlcv.columns:
            return
        volume = self.ohlcv["volume"]
        close = self.ohlcv["close"]
        self.features["volume_roc"] = volume.pct_change(10)
        self.features["obv"] = _obv(close, volume)
        self.features["cmf"] = _cmf(self.ohlcv, 20)
        self.features["volume_sma_ratio"] = volume / (volume.rolling(20).mean() + 1e-8)
        self.features["vwap"] = _vwap(self.ohlcv)

    def add_price_action_features(self) -> None:
        close = self.ohlcv["close"]
        open_ = self.ohlcv["open"]
        high = self.ohlcv["high"]
        low = self.ohlcv["low"]
        self.features["ret_1"] = close.pct_change(1)
        self.features["ret_5"] = close.pct_change(5)
        self.features["ret_10"] = close.pct_change(10)
        self.features["ret_20"] = close.pct_change(20)
        self.features["log_ret"] = np.log(close / close.shift(1))
        self.features["hl_ratio"] = high / (low + 1e-8)
        self.features["co_ratio"] = close / (open_ + 1e-8)
        self.features["true_range"] = _true_range(self.ohlcv)
        shadow_range = (high - low).replace(0, np.nan)
        self.features["upper_shadow_ratio"] = (high - np.maximum(open_, close)) / (shadow_range + 1e-8)
        self.features["lower_shadow_ratio"] = (np.minimum(open_, close) - low) / (shadow_range + 1e-8)

    def add_statistical_features(self) -> None:
        close = self.ohlcv["close"]
        self.features["skew_20"] = close.rolling(20).skew()
        self.features["skew_50"] = close.rolling(50).skew()
        self.features["kurt_20"] = close.rolling(20).kurt()
        self.features["kurt_50"] = close.rolling(50).kurt()
        for lag in range(1, 6):
            self.features[f"autocorr_{lag}"] = close.rolling(50).apply(lambda x: x.autocorr(lag=lag))
        if "volume" in self.ohlcv.columns:
            returns = close.pct_change()
            self.features["ret_vol_corr_20"] = returns.rolling(20).corr(self.ohlcv["volume"])


def create_multistep_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    seq_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    values = features.values
    target_values = target.values
    X, y = [], []
    for i in range(len(features) - seq_len - pred_len + 1):
        X.append(values[i : i + seq_len])
        y.append(target_values[i + seq_len : i + seq_len + pred_len])
    return np.array(X), np.array(y)


def split_time_series(
    features: pd.DataFrame,
    target: pd.Series,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {
        "train": (features.iloc[:train_end], target.iloc[:train_end]),
        "val": (features.iloc[train_end:val_end], target.iloc[train_end:val_end]),
        "test": (features.iloc[val_end:], target.iloc[val_end:]),
    }


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _stoch_rsi(series: pd.Series, period: int = 14, k: int = 3, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    rsi = _rsi(series, period)
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-10)
    stoch_k = stoch.rolling(k).mean()
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = _true_range(df)
    return tr.rolling(period).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    ranges = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = _true_range(df)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    return dx.rolling(period).mean()


def _tema(series: pd.Series, period: int) -> pd.Series:
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def slope(x: np.ndarray) -> float:
        idx = np.arange(len(x))
        if len(x) < 2:
            return np.nan
        coef = np.polyfit(idx, x, 1)
        return coef[0]

    return series.rolling(window).apply(lambda x: slope(x.values), raw=False)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _cmf(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]
    mfm = ((close - low) - (high - close)) / (high - low + 1e-8)
    mfv = mfm * volume
    return mfv.rolling(period).sum() / (volume.rolling(period).sum() + 1e-8)


def _vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"]
    return (typical * volume).cumsum() / (volume.cumsum() + 1e-8)