"""Daily theme screener - picks today's hot stocks for intraday scalping.

Runs once at session start (around 09:00-09:10) and returns a list of
(ticker, asset_cfg) pairs that can be injected into config["universe"]
for the day's scalping run.  All discovered stocks use the same
fixed-stop scalping params so they work with the existing cost gate.

The screener is best-effort: if pykrx is slow or unreachable it logs a
warning and returns [] so the bot carries on with its static universe.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import date, timedelta
from typing import Any, Callable, TypeVar

import pandas as pd

from indicators import atr as atr_indicator
from indicators import ema as ema_indicator

logger = logging.getLogger("bot.screener")

T = TypeVar("T")

#: pykrx's internal HTTP calls carry no timeout of their own. One
#: unresponsive request used to freeze the whole scan -- and since the
#: screener runs on the bot's main loop, that meant freezing the whole bot,
#: with no exception and no log line to explain why. Every pykrx call in
#: this module goes through this wrapper so a single bad request can never
#: hang longer than HTTP_TIMEOUT_SECONDS.
HTTP_TIMEOUT_SECONDS = 8.0


def _with_timeout(fn: Callable[[], T], timeout_seconds: float = HTTP_TIMEOUT_SECONDS) -> T | None:
    """Run *fn* on a daemon thread and give up after *timeout_seconds*.

    Python cannot forcibly kill a thread, so a timed-out call's thread is
    simply abandoned (daemon=True keeps it from blocking process exit) --
    the point is that the CALLER moves on instead of hanging forever.
    """
    result: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result.put(("ok", fn()))
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
            result.put(("error", exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        status, value = result.get(timeout=timeout_seconds)
    except queue.Empty:
        logger.debug("screener: pykrx call timed out after %.0fs", timeout_seconds)
        return None
    if status == "error":
        raise value
    return value

_SCALPING_CFG_TEMPLATE: dict[str, Any] = {
    "enabled": True,
    # 오프닝레인지 브레이크아웃(ORB) -- 09:00-09:15 레인지를 거래량+VWAP+
    # 추세+반등봉강도 4중 확인 후 돌파할 때만 진입. 스크리너가 이미 당일
    # 최고 모멘텀 종목을 고르므로, 첫 15분의 노이즈만 걸러내면 이 종목들의
    # 실제 방향성 있는 움직임을 잡기에 정석 돌파 스타일이 잘 맞는다.
    #
    # volume_mult/min_bar_strength는 2026-08-26 약세장에서 하루 종일 거래 0건
    # (조건 미충족)이 나온 뒤 완화했다 -- 1.5x/0.5는 기관 데스크 기준으로는
    # 정석이지만, 소형주 위주 스크리너 유니버스에는 너무 깐깐해서 방향성 있는
    # 돌파도 자주 걸렀다. 여전히 거래량 동반 + 봉 상단 마감은 요구한다.
    "strategy": "orb",
    "timeframe": "3Min",
    "min_qty": 1,
    "entry_window": ["09:15", "14:30"],  # 레인지(09:00-09:15) 완성 직후부터
    "force_exit_at": "15:10",
    "params": {
        "range_minutes": 15,
        "volume_lookback": 10,
        "volume_mult": 1.2,
        "trend_ema": 21,
        "min_bar_strength": 0.35,
        "stop_pct": 0.017,
        "arm_pct": 0.012,
        "lock_pct": 0.02,
        "peak_trail_pct": 0.003,
        "max_cost_share": 0.35,
    },
}

_PULLBACK_CFG_TEMPLATE: dict[str, Any] = {
    "enabled": True,
    # 눌림목 반등 -- 돌파를 기다리는 ORB와 달리 상승 추세 중 단기 눌림에서
    # 반등을 잡는다. 돌파가 잘 안 나오는 약세/횡보장에서도 진입 기회를 주기
    # 위해 스크리너 유니버스 절반에 ORB와 번갈아 배정한다 (병행 운영).
    # 파라미터는 기존 고정 6종목에서 이미 검증된 값을 그대로 사용.
    "strategy": "pullback_bounce",
    "timeframe": "3Min",
    "min_qty": 1,
    "entry_window": ["09:15", "14:30"],
    "force_exit_at": "15:10",
    "params": {
        "trend_ema": 20,
        "swing_lookback": 7,
        "pullback_bars": 2,
        "pullback_min_pct": 0.005,
        "min_bar_strength": 0.25,
        "stop_pct": 0.017,
        "arm_pct": 0.012,
        "lock_pct": 0.02,
        "peak_trail_pct": 0.003,
        "max_cost_share": 0.35,
    },
}


def _get_market_snapshot(
    date_str: str, market: str, retries: int = 2, retry_delay: float = 2.0
) -> "pd.DataFrame | None":
    """Fetch one market's full snapshot, retrying transient KRX/pykrx blips.

    KRX occasionally answers with an empty body (pykrx then raises a JSON
    decode error) under load or rate limiting; a couple of short retries
    clear most of those without meaningfully slowing the scan down.
    """
    from pykrx import stock as krx

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            df = _with_timeout(lambda: krx.get_market_ohlcv_by_ticker(date_str, market=market))
            if df is None or df.empty:
                return None
            return df
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_delay)
    logger.debug("pykrx snapshot failed %s %s: %s", market, date_str, last_exc)
    return None


def _fetch_history(ticker: str, fromdate: str, todate: str) -> "pd.DataFrame | None":
    try:
        from pykrx import stock as krx
        df = _with_timeout(lambda: krx.get_market_ohlcv_by_date(fromdate, todate, ticker))
        if df is None or df.empty:
            return None
        # pykrx returns Korean column names; normalise them.
        rename = {
            "시가": "open", "고가": "high", "저가": "low",
            "종가": "close", "거래량": "volume",
        }
        df = df.rename(columns=rename)
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return None
        return df
    except Exception as exc:
        logger.debug("pykrx history failed %s: %s", ticker, exc)
        return None


def _ticker_name(ticker: str) -> str:
    try:
        from pykrx import stock as krx
        name = _with_timeout(lambda: krx.get_market_ticker_name(ticker))
        return name or ticker
    except Exception:
        return ticker


class DailyScreener:
    """Picks today's hot scalping candidates from KOSDAQ and/or KOSPI."""

    def __init__(self, config: dict[str, Any]) -> None:
        scr = config.get("screener") or {}
        self.enabled: bool = bool(scr.get("enabled", False))
        self.n_stocks: int = int(scr.get("n_stocks", 5))
        self.min_price: int = int(scr.get("min_price", 2000))
        self.max_price: int = int(scr.get("max_price", 0))  # 0 = no cap
        self.min_atr_pct: float = float(scr.get("min_atr_pct", 0.015))
        self.min_trading_value: float = float(scr.get("min_trading_value_m", 5000)) * 1_000_000
        self.atr_period: int = int((config.get("risk") or {}).get("atr_period", 14))
        # 변동성(ATR)만 크고 실제로는 하락 추세인 종목을 걸러낸다 -- daily_trend.py가
        # 진입 단계에서 쓰는 것과 같은 EMA 기간을 재사용해 두 필터가 어긋나지 않게 한다.
        self.require_uptrend: bool = bool(scr.get("require_uptrend", True))
        self.trend_ema_period: int = int((config.get("daily_trend") or {}).get("ema_period", 20))
        self.markets: list[str] = list(scr.get("markets") or ["KOSDAQ", "KOSPI"])
        self._existing: set[str] = {str(k) for k in (config.get("universe") or {})}
        #: True when every configured market's snapshot fetch failed on the
        #: last scan() call -- distinguishes "pykrx/KRX is down" from
        #: "nothing qualified today", which the caller needs to tell apart to
        #: decide whether a fallback universe is warranted.
        self.last_scan_failed: bool = False

    def scan(self) -> list[tuple[str, dict[str, Any]]]:
        """Return (ticker, asset_cfg) pairs for today's hot scalping stocks.

        Safe to call even when pykrx is unavailable - returns [] on any error.
        """
        self.last_scan_failed = False
        if not self.enabled:
            return []

        today = date.today().isoformat()
        fromdate = (date.today() - timedelta(days=45)).isoformat()

        # (ticker, name, market, atr_pct, trading_value)
        candidates: list[tuple[str, str, str, float, float]] = []
        markets_ok = 0

        for market in self.markets:
            snap = _get_market_snapshot(today, market)
            if snap is None:
                logger.warning("screener: could not fetch %s snapshot for %s", market, today)
                continue
            markets_ok += 1

            col_map = {
                "시가": "open", "고가": "high", "저가": "low", "종가": "close",
                "거래량": "volume", "거래대금": "trading_value",
            }
            snap = snap.rename(columns=col_map)
            if "close" not in snap.columns:
                continue

            n_total = len(snap)
            if "trading_value" in snap.columns:
                snap = snap[snap["trading_value"] >= self.min_trading_value]
            n_after_value = len(snap)
            snap = snap[snap["close"] >= self.min_price]
            if self.max_price > 0:
                snap = snap[snap["close"] <= self.max_price]
            n_after_price = len(snap)
            snap = snap[~snap.index.astype(str).isin(self._existing)]
            n_after_existing = len(snap)
            logger.info(
                "screener: %s 필터 단계 -- 전체 %d -> 거래대금(%.0f억+) %d -> "
                "가격(%d~%d) %d -> 기존제외 %d",
                market, n_total, self.min_trading_value / 100_000_000,
                n_after_value, self.min_price, self.max_price or 999999999,
                n_after_price, n_after_existing,
            )

            if "trading_value" in snap.columns:
                snap = snap.sort_values("trading_value", ascending=False)

            # ATR-check on the top 30 candidates (network-intensive; limit it)
            n_checked = n_no_hist = n_bad_atr = n_low_atr = n_downtrend = n_passed = 0
            for ticker in snap.head(30).index.astype(str):
                n_checked += 1
                hist = _fetch_history(ticker, fromdate, today)
                min_hist = max(self.atr_period, self.trend_ema_period) + 2
                if hist is None or len(hist) < min_hist:
                    n_no_hist += 1
                    continue
                atr_series = atr_indicator(hist, self.atr_period)
                last_atr = atr_series.iloc[-1]
                last_close = float(hist["close"].iloc[-1])
                if last_close <= 0 or pd.isna(last_atr) or float(last_atr) <= 0:
                    n_bad_atr += 1
                    continue
                atr_pct = float(last_atr) / last_close
                if atr_pct < self.min_atr_pct:
                    n_low_atr += 1
                    continue
                if self.require_uptrend:
                    trend_series = ema_indicator(hist["close"], self.trend_ema_period)
                    last_trend = trend_series.iloc[-1]
                    if pd.isna(last_trend) or last_close <= float(last_trend):
                        n_downtrend += 1
                        continue
                n_passed += 1
                tv = 0.0
                if "trading_value" in snap.columns:
                    tv = float(snap.loc[ticker, "trading_value"]) if ticker in snap.index else 0.0
                name = _ticker_name(ticker)
                candidates.append((ticker, name, market, atr_pct, tv))
            if n_checked:
                logger.info(
                    "screener: %s ATR/추세 검사 %d종목 -- 이력부족 %d, ATR계산불가 %d, "
                    "ATR미달(%.1f%%) %d, 하락추세(EMA%d) %d, 통과 %d",
                    market, n_checked, n_no_hist, n_bad_atr,
                    self.min_atr_pct * 100, n_low_atr,
                    self.trend_ema_period, n_downtrend, n_passed,
                )

        if markets_ok == 0 and self.markets:
            self.last_scan_failed = True
            logger.warning(
                "screener: every configured market failed to fetch -- "
                "treating this as an infra failure, not zero matches"
            )

        candidates.sort(key=lambda x: x[3], reverse=True)

        results: list[tuple[str, dict[str, Any]]] = []
        for i, (ticker, name, market, atr_pct, tv) in enumerate(candidates[: self.n_stocks]):
            # ORB(돌파 대기)와 눌림목(하락 없이도 진입) 을 절반씩 번갈아 배정 --
            # 돌파가 잘 안 나오는 장에서도 눌림목 쪽에서 기회를 잡을 수 있게.
            template = _SCALPING_CFG_TEMPLATE if i % 2 == 0 else _PULLBACK_CFG_TEMPLATE
            cfg: dict[str, Any] = {
                **template,
                "name": name,
                "market": market,
                "params": dict(template["params"]),
                "_screener": True,
            }
            logger.info(
                "screener: %s %s [%s]  ATR %.1f%%  TV KRW%.0fM",
                ticker, name, template["strategy"], atr_pct * 100, tv / 1_000_000,
            )
            results.append((ticker, cfg))

        return results
