"""KRX market data.

Resolution order for every request:

1. a CSV cache under ``data/cache/`` - fast, offline, reproducible;
2. a live source - **pykrx** for daily bars, the **broker's chart API** for
   intraday bars (pykrx does not serve minute data);
3. a deterministic synthetic generator.

Step 3 exists so the whole pipeline - backtest, sizing, reporting, tests - can
be exercised with no credentials and no network. Whenever synthetic bars are
used the caller is told loudly and every report that consumed them is stamped
``SYNTHETIC DATA``.

``pykrx`` is imported lazily inside the fetch helper so the rest of the code
base, and the whole test suite, runs with it absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from market.calendar import KST, KrxCalendar
from market.rules import KOSDAQ, KOSPI

logger = logging.getLogger("bot.data")

Source = Literal["cache", "pykrx", "broker", "synthetic"]

#: Supported timeframe labels mapped to their bar duration.
TIMEFRAMES: dict[str, timedelta] = {
    "1Min": timedelta(minutes=1),
    "3Min": timedelta(minutes=3),
    "5Min": timedelta(minutes=5),
    "10Min": timedelta(minutes=10),
    "15Min": timedelta(minutes=15),
    "30Min": timedelta(minutes=30),
    "60Min": timedelta(hours=1),
    "1Day": timedelta(days=1),
}

#: Rough per-bar volatility for the synthetic generator. Korean small/mid caps
#: are livelier than US index ETFs, so these are wider than the US bot's.
_SYNTHETIC_SIGMA = {
    "1Min": 0.0012,
    "3Min": 0.002,
    "5Min": 0.0026,
    "10Min": 0.0034,
    "15Min": 0.004,
    "30Min": 0.006,
    "60Min": 0.009,
    "1Day": 0.030,
}


#: Market labels seen from Kiwoom, mapped to the two boards. Anything not
#: listed here resolves to None rather than a guess -- the market tag selects
#: the transaction tax, so a wrong one is worse than an unverified one.
_KOSPI_LABELS = {KOSPI, "0", "1", "KRX", "STK", "코스피", "유가증권", "유가"}
_KOSDAQ_LABELS = {KOSDAQ, "10", "KSQ", "코스닥"}


class IntradayProvider(Protocol):
    """Anything that can serve minute bars - in practice, the broker."""

    def get_chart(
        self,
        code: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        max_rows: int | None = None,
    ) -> pd.DataFrame | None: ...


@dataclass(frozen=True)
class BarSet:
    """OHLCV bars plus the provenance of those bars."""

    code: str
    timeframe: str
    bars: pd.DataFrame
    source: Source
    market: str = KOSPI

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


def is_intraday(timeframe: str) -> bool:
    return timeframe_delta(timeframe) < timedelta(days=1)


def bars_per_year(timeframe: str) -> float:
    """Approximate bars in a calendar year - used to annualise."""
    if not is_intraday(timeframe):
        return 245.0
    session_seconds = 6.5 * 3600  # 09:00-15:30
    return 245.0 * session_seconds / timeframe_delta(timeframe).total_seconds()


def months_to_start(months: int, end: datetime | None = None) -> datetime:
    end = end or datetime.now(KST)
    return end - timedelta(days=int(round(months * 30.44)))


def warmup_start(start: datetime, timeframe: str, warmup_bars: int) -> datetime:
    """Shift *start* back far enough to warm up a ``warmup_bars`` indicator.

    Without this a six-month backtest of a 200-day EMA evaluates nothing: 200
    trading days is roughly ten calendar months, so every bar in the requested
    window would still be inside the warm-up period. Fetching the run-up
    separately lets the requested window actually be traded.
    """
    if warmup_bars <= 0:
        return start
    if is_intraday(timeframe):
        bars_per_session = (6.5 * 3600) / timeframe_delta(timeframe).total_seconds()
        sessions = warmup_bars / max(bars_per_session, 1e-9)
    else:
        sessions = warmup_bars
    # ~245 trading days per 365 calendar days, plus a margin for holidays.
    calendar_days = sessions * (365.0 / 245.0) + 10
    return start - timedelta(days=int(calendar_days))


# --------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------


def _session_index(
    start: datetime,
    end: datetime,
    timeframe: str,
    calendar: KrxCalendar,
) -> pd.DatetimeIndex:
    """Bar timestamps on the KRX session grid (09:00-15:30 KST)."""
    days = calendar.business_days(start.date(), end.date())
    if not days:
        return pd.DatetimeIndex([], tz=KST, name="timestamp")

    if not is_intraday(timeframe):
        stamps = [datetime.combine(d, datetime.min.time(), tzinfo=KST) for d in days]
        return pd.DatetimeIndex(stamps, name="timestamp")

    delta = timeframe_delta(timeframe)
    stamps: list[datetime] = []
    for day in days:
        cursor = datetime.combine(day, datetime.min.time(), tzinfo=KST).replace(hour=9)
        close = cursor.replace(hour=15, minute=30)
        while cursor < close:
            stamps.append(cursor)
            cursor += delta
    return pd.DatetimeIndex(stamps, name="timestamp")


def _code_seed(code: str, timeframe: str) -> int:
    return abs(hash((code, timeframe))) % (2**31)


def generate_synthetic_bars(
    code: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    calendar: KrxCalendar,
    start_price: float = 20_000.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Deterministic OHLCV bars on the KRX session grid.

    The output is *plausible*, not real: trends, mean-reverting stretches and
    volume spikes so every strategy can fire. It makes no claim to resemble the
    actual security.
    """
    index = _session_index(start, end, timeframe, calendar)
    n = len(index)
    if n == 0:
        return empty_frame()

    rng = np.random.default_rng(seed if seed is not None else _code_seed(code, timeframe))
    sigma = _SYNTHETIC_SIGMA.get(timeframe, 0.01)

    phase = np.linspace(0.0, 6.0 * np.pi, n)
    drift = 0.35 * sigma * np.sin(phase) + 0.02 * sigma * np.sin(phase * 7.3)
    log_returns = drift + rng.normal(0.0, sigma, n)
    close = start_price * np.exp(np.cumsum(log_returns))

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    wick = np.abs(rng.normal(0.0, sigma * 0.8, n))
    high = np.maximum(open_, close) * (1.0 + wick)
    low = np.minimum(open_, close) * (1.0 - wick)

    volume = rng.lognormal(mean=np.log(400_000), sigma=0.45, size=n)
    volume *= 1.0 + 1.5 * np.abs(log_returns) / sigma

    frame = pd.DataFrame(
        {
            "open": np.round(open_),
            "high": np.round(high),
            "low": np.round(low),
            "close": np.round(close),
            "volume": np.round(volume),
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in ("open", "high", "low", "close", "volume")},
        index=pd.DatetimeIndex([], tz=KST, name="timestamp"),
    )


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def _cache_path(cache_dir: Path, code: str, timeframe: str) -> Path:
    return cache_dir / f"{code}_{timeframe}.csv"


def read_cache(cache_dir: Path, code: str, timeframe: str) -> pd.DataFrame | None:
    path = _cache_path(cache_dir, code, timeframe)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError) as exc:
        logger.warning("cache read failed for %s %s: %s", code, timeframe, exc)
        return None
    if frame.empty:
        return None
    frame.index = (
        frame.index.tz_localize(KST) if frame.index.tz is None else frame.index.tz_convert(KST)
    )
    frame.index.name = "timestamp"
    return frame.sort_index()


def write_cache(cache_dir: Path, code: str, timeframe: str, frame: pd.DataFrame) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(_cache_path(cache_dir, code, timeframe))
    except OSError as exc:
        logger.warning("cache write failed for %s %s: %s", code, timeframe, exc)


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


class MarketData:
    """Fetches KRX bars, with cache and synthetic fallbacks."""

    def __init__(
        self,
        calendar: KrxCalendar | None = None,
        cache_dir: str | Path = "data/cache",
        intraday_provider: IntradayProvider | None = None,
        allow_synthetic: bool = True,
        allow_network: bool = True,
    ) -> None:
        self.calendar = calendar or KrxCalendar()
        self.cache_dir = Path(cache_dir)
        self.intraday_provider = intraday_provider
        self.allow_synthetic = allow_synthetic
        self.allow_network = allow_network
        self._used_synthetic = False

    @property
    def used_synthetic(self) -> bool:
        return self._used_synthetic

    def get_bars(
        self,
        code: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        lookback_bars: int | None = None,
        market: str = KOSPI,
        use_cache: bool = True,
        synthetic_start_price: float = 20_000.0,
    ) -> BarSet:
        timeframe_delta(timeframe)  # validates the label
        end = end or datetime.now(KST)
        if start is None:
            lookback_bars = lookback_bars or 300
            if is_intraday(timeframe):
                # Only 6.5 hours of each day carry bars, so ask wide and trim.
                span = timeframe_delta(timeframe) * lookback_bars
                start = end - span * 4
            else:
                start = end - timedelta(days=int(lookback_bars * 1.5) + 10)

        if use_cache:
            cached = read_cache(self.cache_dir, code, timeframe)
            if cached is not None:
                window = cached.loc[(cached.index >= start) & (cached.index <= end)]
                # Enough bars is not the same as current bars. A cache that
                # satisfied the count check was short-circuiting every live
                # fetch, so a daily strategy kept deciding on bars that ended
                # days earlier while reporting them as good data.
                if len(window) >= (lookback_bars or 1) and self.is_current(
                    window, timeframe, end
                ):
                    return BarSet(code, timeframe, _tail(window, lookback_bars), "cache", market)

        frame, source = self._fetch_live(
            code, timeframe, start, end, market, lookback_bars=lookback_bars
        )
        if frame is not None and not frame.empty:
            if use_cache:
                write_cache(self.cache_dir, code, timeframe, frame)
            return BarSet(code, timeframe, _tail(frame, lookback_bars), source, market)

        if not self.allow_synthetic:
            raise RuntimeError(
                f"no market data available for {code} {timeframe} "
                "(no cache, no live source, synthetic fallback disabled)"
            )

        self._used_synthetic = True
        logger.warning(
            "SYNTHETIC DATA in use for %s %s - results are illustrative only", code, timeframe
        )
        frame = generate_synthetic_bars(
            code, timeframe, start, end, self.calendar, synthetic_start_price
        )
        return BarSet(code, timeframe, _tail(frame, lookback_bars), "synthetic", market)

    # -- live sources ------------------------------------------------------

    def is_current(
        self, frame: pd.DataFrame, timeframe: str, end: datetime | None = None
    ) -> bool:
        """Do these bars run up to the most recent bar that should exist?

        Daily bars must reach today once continuous trading has begun, because
        the loop reads the last close as the current price -- yesterday's close
        prices a stop against a level the stock has already left. Outside the
        session the last completed day is enough; today's bar does not exist
        yet before 09:00, and demanding it would block every pre-open run.

        Intraday bars get a two-interval grace, which covers the gap between
        one bar closing and the next being published.
        """
        if frame is None or frame.empty:
            return False
        end = end or datetime.now(KST)
        last = frame.index[-1]
        if getattr(last, "tzinfo", None) is None:
            last = last.tz_localize(KST) if hasattr(last, "tz_localize") else last
        if is_intraday(timeframe):
            try:
                return (end - last) <= timeframe_delta(timeframe) * 2
            except TypeError:  # naive/aware mismatch we could not repair
                return True
        if self.calendar.is_open(end):
            expected = end.date()
        else:
            expected = self.calendar.previous_business_day(end.date())
        return last.date() >= expected

    def _fetch_live(
        self,
        code: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        market: str,
        lookback_bars: int | None = None,
    ) -> tuple[pd.DataFrame | None, Source]:
        if not self.allow_network:
            return None, "synthetic"
        # A generous multiple of what the caller asked for: enough that the
        # window is never short, small enough that a minute chart stops after
        # a page or two instead of pulling months nothing will read.
        max_rows = int(lookback_bars * 3) if lookback_bars else None
        if is_intraday(timeframe):
            return self._fetch_broker(code, timeframe, start, end, max_rows), "broker"
        frame = self._fetch_pykrx_daily(code, start, end)
        if frame is not None and not frame.empty:
            return frame, "pykrx"
        # KRX's own endpoints go down, and pykrx then returns nothing at all.
        # Kiwoom serves daily bars over the same chart call the intraday
        # stocks already use, so there is a second source to try before
        # falling back to a cache that may be stale or to invented numbers.
        logger.info("pykrx returned nothing for %s; trying the broker chart", code)
        return self._fetch_broker(code, "1Day", start, end, max_rows), "broker"

    def _fetch_broker(
        self,
        code: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        max_rows: int | None = None,
    ) -> pd.DataFrame | None:
        """Bars from the broker's chart call.

        Minute bars have no other source -- pykrx does not serve them -- and
        daily bars come through here when pykrx is unreachable.
        """
        if self.intraday_provider is None:
            logger.info(
                "no broker configured; %s %s cannot be fetched live", code, timeframe
            )
            return None
        try:
            frame = self.intraday_provider.get_chart(
                code, timeframe, start, end, max_rows=max_rows
            )
        except Exception as exc:
            logger.warning("broker chart fetch failed for %s %s: %s", code, timeframe, exc)
            return None
        return _normalise(frame)

    def _fetch_pykrx_daily(
        self, code: str, start: datetime, end: datetime
    ) -> pd.DataFrame | None:
        try:
            # Lazy import: pykrx stays optional for tests and offline runs.
            stock = _import_pykrx_stock()
        except ImportError:
            logger.info("pykrx is not installed; skipping daily fetch for %s", code)
            return None
        try:
            frame = stock.get_market_ohlcv(
                start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code
            )
        except Exception as exc:
            logger.warning("pykrx fetch failed for %s: %s", code, exc)
            return None
        if frame is None or frame.empty:
            return None
        frame = frame.rename(
            columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
        )
        return _normalise(frame)

    def _broker_stock_info(self, code: str):
        """What the broker knows about a listing, if a broker is connected.

        KRX's own ticker endpoints have been returning non-JSON, which leaves
        pykrx unable to resolve a single name -- and with no name resolved,
        the config-vs-listing comparison that catches a wrong market tag never
        runs. Kiwoom answers the same question over an endpoint the connection
        check already exercises.
        """
        provider = self.intraday_provider
        if provider is None or not hasattr(provider, "get_stock_info"):
            return None
        try:
            return provider.get_stock_info(code)
        except Exception as exc:
            logger.debug("broker stock info failed for %s: %s", code, exc)
            return None

    def get_name(self, code: str) -> str | None:
        """Official listing name from pykrx, or failing that the broker."""
        if not self.allow_network:
            return None
        try:
            stock = _import_pykrx_stock()

            name = stock.get_market_ticker_name(code)
            if name:
                return str(name)
        except Exception as exc:
            logger.debug("name lookup failed for %s: %s", code, exc)
        info = self._broker_stock_info(code)
        if info is None or not info.name:
            return None
        return str(info.name).strip() or None

    def get_market(self, code: str) -> str | None:
        """KOSPI or KOSDAQ from pykrx, or failing that the broker."""
        if not self.allow_network:
            return None
        try:
            stock = _import_pykrx_stock()

            today = date.today().strftime("%Y%m%d")
            for name in (KOSPI, KOSDAQ):
                if code in set(stock.get_market_ticker_list(today, market=name)):
                    return name
        except Exception as exc:
            logger.debug("market lookup failed for %s: %s", code, exc)
        info = self._broker_stock_info(code)
        if info is None:
            return None
        # market_raw, never `market`: the latter defaults to KOSPI when the
        # field is absent, which would confirm a KOSPI tag on every stock and
        # flag every KOSDAQ one as a mismatch. That is what it did.
        raw = str(getattr(info, "market_raw", "") or "").strip().upper()
        if not raw:
            return None
        if raw in _KOSPI_LABELS:
            return KOSPI
        if raw in _KOSDAQ_LABELS:
            return KOSDAQ
        logger.info(
            "broker reports market %r for %s, which maps to neither board; "
            "leaving the configured tag alone",
            raw,
            code,
        )
        return None


def _import_pykrx_stock():
    """Import ``pykrx.stock``, silencing its import-time login notice.

    pykrx prints "KRX 로그인 실패: KRX_ID 또는 KRX_PW 환경 변수가 설정되지
    않았습니다." to stdout the first time it is imported without those
    variables. It reads like a failure but is not one: ``website/comm/webio.py``
    falls back to an ordinary anonymous request when no session exists, which is
    how pykrx has always fetched public KRX data. Only the import-time banner is
    suppressed - real fetch errors still surface through the caller.
    """
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        from pykrx import stock

    return stock


def _normalise(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]
    if len(keep) < 5:
        return None
    out = frame[keep].copy()
    out.index = pd.to_datetime(out.index)
    out.index = (
        out.index.tz_localize(KST) if out.index.tz is None else out.index.tz_convert(KST)
    )
    out.index.name = "timestamp"
    # KRX publishes suspended days as all-zero rows; they are not tradable bars.
    out = out[out["close"] > 0]
    return out.sort_index()


def _tail(frame: pd.DataFrame, lookback_bars: int | None) -> pd.DataFrame:
    if lookback_bars is None or len(frame) <= lookback_bars:
        return frame
    return frame.iloc[-lookback_bars:]
