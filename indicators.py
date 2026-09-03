"""Technical indicators.

Every function here is a pure transformation of an OHLCV frame into a Series
aligned on the same index.  The value at position ``i`` is computed from bars
``0..i`` only - never from bar ``i+1``.  Strategies that need a *prior* level
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


def bollinger_bands(
    series: pd.Series, period: int = 20, mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: (middle, upper, lower).

    Middle is the plain SMA(period) -- the same 20-bar average professionals
    read as "the trend" on its own. The bands widen and narrow with recent
    volatility (rolling std), so price riding the upper band is extended
    relative to its *own* recent range, not just up in absolute terms; a dip
    to the lower band inside an uptrend is the classic "buy the band" entry.
    """
    mid = sma(series, period)
    std = rolling_std(series, period)
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower


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


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """Cumulative volume: added on an up close, subtracted on a down close.

    A rising OBV over several bars reads as net accumulation even on days
    that individually look quiet -- the multi-bar counterpart to a single
    day's close-strength/volume check.
    """
    validate_ohlcv(df)
    direction = df["close"].diff().apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    return (direction * df["volume"]).cumsum()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder smoothing (0-100).

    >70 is the conventional "overbought" line -- a bounce entered up there is
    chasing a move that has already used up most of its room. <30 is
    "oversold", the mirror case for a short.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    result = 100 - (100 / (1 + rs))
    # avg_loss == 0 and avg_gain > 0: every move was up -- maximally strong.
    # avg_loss == 0 and avg_gain == 0 too (a flat window): no moves at all,
    # which is neutral, not "overbought".
    result = result.where(avg_loss != 0, 100.0)
    return result.where((avg_loss != 0) | (avg_gain != 0), 50.0)


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line and histogram (line - signal).

    The histogram turning from negative to positive is the standard momentum
    confirmation professionals look for on top of a price-action entry -- it
    says the *rate* of the move is accelerating, not just that price ticked
    up one bar.
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def nearest_resistance(df: pd.DataFrame, lookback: int, price: float) -> float | None:
    """Highest high in the last *lookback* bars that sits above *price*.

    A simple, defensible read of "resistance": the last place sellers showed
    up in size. Buying within a hair of it means the very next tick can stall
    the trade -- professionals give a breakout room, not a fresh high.
    """
    recent_high = df["high"].tail(lookback)
    above = recent_high[recent_high > price]
    if above.empty:
        return None
    return float(above.max())


def bar_strength(df: pd.DataFrame) -> pd.Series:
    """Fraction of each bar's range where the close landed, 0=low 1=high.

    A value >= 0.5 means the bar closed in the upper half of its range,
    which is a sign that buyers controlled the bar regardless of volume.
    """
    validate_ohlcv(df)
    rng = df["high"] - df["low"]
    strength = (df["close"] - df["low"]) / rng.replace(0, pd.NA)
    return strength.fillna(0.5)


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
