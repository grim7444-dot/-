"""Market data access.

Resolution order for every request:

1. a CSV cache under ``data/cache/`` (fast, offline, reproducible),
2. the Alpaca historical data API (requires credentials from ``.env``),
3. a deterministic synthetic generator.

Step 3 exists so the whole pipeline — backtest, risk sizing, reporting,
tests — can be exercised without credentials and without a network.  Whenever
synthetic bars are used the caller is told loudly, and every report that
consumed them is stamped ``SYNTHETIC DATA``.

The ``alpaca`` SDK is imported lazily inside the fetch helper so that the rest
of the code base (and the whole test suite) runs with the SDK absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from settings import Credentials, mask_text

logger = logging.getLogger("bot.data")

Source = Literal["cache", "alpaca", "synthetic"]

#: Supported timeframe labels mapped to their bar duration.
TIMEFRAMES: dict[str, timedelta] = {
    "15Min": timedelta(minutes=15),
    "1Hour": timedelta(hours=1),
    "4Hour": timedelta(hours=4),
}

#: Rough per-bar volatility used by the synthetic generator, by timeframe.
_SYNTHETIC_SIGMA = {"15Min": 0.0012, "1Hour": 0.0035, "4Hour": 0.0070}

_SYNTHETIC_START_PRICE = {
    "SPY": 520.0,
    "QQQ": 440.0,
    "BTC/USD": 62000.0,
    "GLD": 215.0,
    "USO": 78.0,
}


@dataclass(frozen=True)
class BarSet:
    """OHLCV bars plus the provenance of those bars."""

    symbol: str
    timeframe: str
    bars: pd.DataFrame
    source: Source

    @property
    def synthetic(self) -> bool:
        return self.source == "synthetic"

    def __len__(self) -> int:
        return len(self.bars)


def timeframe_delta(timeframe: str) -> timedelta:
    try:
        return TIMEFRAMES[timeframe]
    except KeyError:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; expected one of {sorted(TIMEFRAMES)}"
        ) from None


def bars_per_year(timeframe: str, asset_class: str) -> float:
    """Approximate number of bars in a calendar year — used to annualise."""
    delta = timeframe_delta(timeframe)
    if asset_class == "crypto":
        seconds = 365 * 24 * 3600
    else:
        # 252 trading days of 6.5 regular-session hours.
        seconds = 252 * 6.5 * 3600
    return seconds / delta.total_seconds()


# --------------------------------------------------------------------------
# Session-aware timestamp grid
# --------------------------------------------------------------------------


def _session_index(
    start: datetime,
    end: datetime,
    timeframe: str,
    asset_class: str,
) -> pd.DatetimeIndex:
    """Build the bar timestamps between *start* and *end*.

    Crypto is continuous.  Equities are restricted to US trading hours, which
    keeps the 15-minute SPY/QQQ series honest about overnight gaps.

    The window depends on the bar size, matching how Alpaca aggregates: minute
    bars cover the regular session (09:30–16:00 ET), while hourly and larger
    buckets span the extended session (04:00–20:00 ET).  That distinction
    matters — on a regular-session-only grid a 4-hour series yields barely one
    bar a day, and a 200-period EMA would need years of history to warm up.
    """
    delta = timeframe_delta(timeframe)
    freq = pd.Timedelta(delta)
    index = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    if asset_class == "crypto":
        return index

    local = index.tz_convert("America/New_York")
    weekday = local.weekday < 5
    minutes = local.hour * 60 + local.minute
    if delta >= timedelta(hours=1):
        in_session = (minutes >= 4 * 60) & (minutes < 20 * 60)
    else:
        in_session = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    return index[weekday & in_session]


# --------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------


def _symbol_seed(symbol: str, timeframe: str) -> int:
    """Stable seed so a given symbol/timeframe always yields the same series."""
    return abs(hash((symbol, timeframe))) % (2**31)


def generate_synthetic_bars(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    asset_class: str = "us_equity",
    seed: int | None = None,
) -> pd.DataFrame:
    """Deterministic geometric-Brownian-motion OHLCV bars.

    The output is *plausible*, not real: it has trends, mean-reverting
    stretches and volume spikes so every strategy can fire, but no claim is
    made that it resembles the actual instrument.
    """
    index = _session_index(start, end, timeframe, asset_class)
    n = len(index)
    if n == 0:
        return _empty_frame()

    rng = np.random.default_rng(seed if seed is not None else _symbol_seed(symbol, timeframe))
    sigma = _SYNTHETIC_SIGMA.get(timeframe, 0.003)
    if asset_class == "crypto":
        sigma *= 2.5

    # A slow sinusoidal drift creates regime changes, so trend-following and
    # mean-reversion strategies both see something to trade.
    phase = np.linspace(0.0, 6.0 * np.pi, n)
    drift = 0.35 * sigma * np.sin(phase) + 0.02 * sigma * np.sin(phase * 7.3)
    shocks = rng.normal(loc=0.0, scale=sigma, size=n)
    log_returns = drift + shocks

    start_price = _SYNTHETIC_START_PRICE.get(symbol, 100.0)
    close = start_price * np.exp(np.cumsum(log_returns))

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    wick = np.abs(rng.normal(0.0, sigma * 0.8, size=n))
    high = np.maximum(open_, close) * (1.0 + wick)
    low = np.minimum(open_, close) * (1.0 - wick)

    base_volume = 1_500_000 if asset_class != "crypto" else 900
    volume = rng.lognormal(mean=np.log(base_volume), sigma=0.35, size=n)
    # Volume expands with the size of the move, so breakout volume filters are
    # meaningful rather than random.
    volume *= 1.0 + 6.0 * np.abs(log_returns) / sigma * 0.25

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.round(volume, 4),
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


def _empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in ("open", "high", "low", "close", "volume")},
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
    )
    return frame


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def _cache_path(cache_dir: Path, symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "-")
    return cache_dir / f"{safe}_{timeframe}.csv"


def read_cache(cache_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame | None:
    path = _cache_path(cache_dir, symbol, timeframe)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError) as exc:
        logger.warning("cache read failed for %s %s: %s", symbol, timeframe, exc)
        return None
    if frame.empty:
        return None
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame.index.name = "timestamp"
    return frame.sort_index()


def write_cache(cache_dir: Path, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(_cache_path(cache_dir, symbol, timeframe))
    except OSError as exc:
        logger.warning("cache write failed for %s %s: %s", symbol, timeframe, exc)


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


class MarketData:
    """Fetches bars for the bot, with cache and synthetic fallbacks."""

    def __init__(
        self,
        credentials: Credentials | None = None,
        cache_dir: str | Path = "data/cache",
        allow_synthetic: bool = True,
        allow_network: bool = True,
    ) -> None:
        self.credentials = credentials
        self.cache_dir = Path(cache_dir)
        self.allow_synthetic = allow_synthetic
        self.allow_network = allow_network
        self._used_synthetic = False

    @property
    def used_synthetic(self) -> bool:
        """True once any request in this session fell back to synthetic bars."""
        return self._used_synthetic

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        lookback_bars: int | None = None,
        asset_class: str = "us_equity",
        use_cache: bool = True,
    ) -> BarSet:
        timeframe_delta(timeframe)  # validates the label
        end = end or datetime.now(timezone.utc)
        if start is None:
            if lookback_bars is None:
                lookback_bars = 500
            # Equity sessions only cover ~27% of the wall clock, so ask for a
            # generous window and trim afterwards.
            span = timeframe_delta(timeframe) * lookback_bars
            start = end - (span if asset_class == "crypto" else span * 4)

        if use_cache:
            cached = read_cache(self.cache_dir, symbol, timeframe)
            if cached is not None:
                window = cached.loc[(cached.index >= start) & (cached.index <= end)]
                if len(window) >= (lookback_bars or 1):
                    logger.debug("cache hit for %s %s (%d bars)", symbol, timeframe, len(window))
                    return BarSet(symbol, timeframe, _tail(window, lookback_bars), "cache")

        frame = self._fetch_alpaca(symbol, timeframe, start, end, asset_class)
        if frame is not None and not frame.empty:
            if use_cache:
                write_cache(self.cache_dir, symbol, timeframe, frame)
            return BarSet(symbol, timeframe, _tail(frame, lookback_bars), "alpaca")

        if not self.allow_synthetic:
            raise RuntimeError(
                f"no market data available for {symbol} {timeframe} "
                "(no cache, no API credentials, synthetic fallback disabled)"
            )

        self._used_synthetic = True
        logger.warning(
            "SYNTHETIC DATA in use for %s %s — results are illustrative only",
            symbol,
            timeframe,
        )
        frame = generate_synthetic_bars(symbol, timeframe, start, end, asset_class)
        return BarSet(symbol, timeframe, _tail(frame, lookback_bars), "synthetic")

    # -- Alpaca ------------------------------------------------------------

    def _fetch_alpaca(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        asset_class: str,
    ) -> pd.DataFrame | None:
        if not self.allow_network:
            return None
        if self.credentials is None or not self.credentials.has_alpaca:
            logger.info(
                "no Alpaca credentials available; skipping API fetch for %s", symbol
            )
            return None
        try:
            return self._fetch_alpaca_inner(symbol, timeframe, start, end, asset_class)
        except Exception as exc:  # network, auth, rate limit, SDK mismatch
            # mask_text keeps a credential out of the message if the SDK echoed
            # one back inside an error string.
            logger.warning(
                "Alpaca data fetch failed for %s %s: %s",
                symbol,
                timeframe,
                mask_text(str(exc)),
            )
            return None

    def _fetch_alpaca_inner(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        asset_class: str,
    ) -> pd.DataFrame | None:
        # Lazy import: the SDK is optional for tests and offline runs.
        from alpaca.data.historical import (  # type: ignore import-not-found
            CryptoHistoricalDataClient,
            StockHistoricalDataClient,
        )
        from alpaca.data.requests import (  # type: ignore import-not-found
            CryptoBarsRequest,
            StockBarsRequest,
        )
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore

        tf_map = {
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
        }
        tf = tf_map[timeframe]
        key = self.credentials.api_key.reveal()
        secret = self.credentials.secret_key.reveal()

        if asset_class == "crypto":
            client = CryptoHistoricalDataClient(key, secret)
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol, timeframe=tf, start=start, end=end
            )
            response = client.get_crypto_bars(request)
        else:
            client = StockHistoricalDataClient(key, secret)
            request = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=tf, start=start, end=end
            )
            response = client.get_stock_bars(request)

        frame = response.df
        if frame is None or frame.empty:
            return None
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.xs(symbol, level="symbol")
        frame = frame.rename(columns=str.lower)
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]
        frame = frame[keep].copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame.index.name = "timestamp"
        return frame.sort_index()


def _tail(frame: pd.DataFrame, lookback_bars: int | None) -> pd.DataFrame:
    if lookback_bars is None or len(frame) <= lookback_bars:
        return frame
    return frame.iloc[-lookback_bars:]


def months_to_start(months: int, end: datetime | None = None) -> datetime:
    """Approximate calendar start date *months* before *end*."""
    end = end or datetime.now(timezone.utc)
    return end - timedelta(days=int(round(months * 30.44)))
