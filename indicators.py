"""Technical indicators.

Every function here is a pure transformation of an OHLCV frame into a Series
aligned on the same index.  The value at position ``i`` is computed from bars
``0..i`` only — never from bar ``i+1``.  Strategies that need a *prior* level
(for example "did we break the previous 20-bar high?") must call ``.shift(1)``
explicitly at the call site so the intent is visible in the strategy code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Check that *df* looks like an OHLCV frame and return it unchanged."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV frame index must be sorted ascending")
    return df


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average over *period* bars."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (``adjust=False``, classic recursive form)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rolling_std(series: pd.Series, period: int) -> pd.Series:
    """Rolling sample standard deviation over *period* bars."""
    return series.rolling(window=period, min_periods=period).std(ddof=1)


def zscore(series: pd.Series, period: int) -> pd.Series:
    """Distance from the rolling mean expressed in rolling standard deviations."""
    mean = sma(series, period)
    std = rolling_std(series, period)
    # A flat window has zero dispersion; there is no meaningful z-score there.
    std = std.where(std > 0)
    return (series - mean) / std


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's true range."""
    validate_ohlcv(df)
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range, Wilder smoothing (equivalent to ``alpha=1/period``)."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rolling_max(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).max()


def rolling_min(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).min()


def rolling_mean_volume(df: pd.DataFrame, period: int) -> pd.Series:
    validate_ohlcv(df)
    return df["volume"].rolling(window=period, min_periods=period).mean()


def last_valid(series: pd.Series) -> float | None:
    """Return the last non-NaN value of *series*, or ``None`` if there is none."""
    if series is None or len(series) == 0:
        return None
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    value = float(cleaned.iloc[-1])
    if not np.isfinite(value):
        return None
    return value
