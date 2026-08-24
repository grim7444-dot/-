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
from datetime import date, timedelta
from typing import Any

import pandas as pd

from indicators import atr as atr_indicator

logger = logging.getLogger("bot.screener")

_SCALPING_CFG_TEMPLATE: dict[str, Any] = {
    "enabled": True,
    "strategy": "pullback_bounce",
    "timeframe": "3Min",
    "min_qty": 1,
    "entry_window": ["09:00", "11:00"],
    "force_exit_at": "15:10",
    "params": {
        "trend_ema": 20,
        "swing_lookback": 10,
        "pullback_bars": 3,
        "pullback_min_pct": 0.008,
        "min_bar_strength": 0.3,
        "stop_pct": 0.02,
        "arm_pct": 0.02,
        "lock_pct": 0.03,
        "cap_pct": 0.05,
        "peak_trail_pct": 0.02,
        "max_cost_share": 0.35,
    },
}


def _get_market_snapshot(date_str: str, market: str) -> "pd.DataFrame | None":
    try:
        from pykrx import stock as krx
        df = krx.get_market_ohlcv_by_ticker(date_str, market=market)
        if df is None or df.empty:
            return None
        return df
    except Exception as exc:
        logger.debug("pykrx snapshot failed %s %s: %s", market, date_str, exc)
        return None


def _fetch_history(ticker: str, fromdate: str, todate: str) -> "pd.DataFrame | None":
    try:
        from pykrx import stock as krx
        df = krx.get_market_ohlcv_by_date(fromdate, todate, ticker)
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
        return krx.get_market_ticker_name(ticker) or ticker
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
        self.markets: list[str] = list(scr.get("markets") or ["KOSDAQ", "KOSPI"])
        self._existing: set[str] = {str(k) for k in (config.get("universe") or {})}

    def scan(self) -> list[tuple[str, dict[str, Any]]]:
        """Return (ticker, asset_cfg) pairs for today's hot scalping stocks.

        Safe to call even when pykrx is unavailable - returns [] on any error.
        """
        if not self.enabled:
            return []

        today = date.today().isoformat()
        fromdate = (date.today() - timedelta(days=45)).isoformat()

        # (ticker, name, market, atr_pct, trading_value)
        candidates: list[tuple[str, str, str, float, float]] = []

        for market in self.markets:
            snap = _get_market_snapshot(today, market)
            if snap is None:
                logger.warning("screener: could not fetch %s snapshot for %s", market, today)
                continue

            col_map = {
                "시가": "open", "고가": "high", "저가": "low", "종가": "close",
                "거래량": "volume", "거래대금": "trading_value",
            }
            snap = snap.rename(columns=col_map)
            if "close" not in snap.columns:
                continue

            if "trading_value" in snap.columns:
                snap = snap[snap["trading_value"] >= self.min_trading_value]
            snap = snap[snap["close"] >= self.min_price]
            if self.max_price > 0:
                snap = snap[snap["close"] <= self.max_price]
            snap = snap[~snap.index.astype(str).isin(self._existing)]

            if "trading_value" in snap.columns:
                snap = snap.sort_values("trading_value", ascending=False)

            # ATR-check on the top 30 candidates (network-intensive; limit it)
            for ticker in snap.head(30).index.astype(str):
                hist = _fetch_history(ticker, fromdate, today)
                if hist is None or len(hist) < self.atr_period + 2:
                    continue
                atr_series = atr_indicator(hist, self.atr_period)
                last_atr = atr_series.iloc[-1]
                last_close = float(hist["close"].iloc[-1])
                if last_close <= 0 or pd.isna(last_atr) or float(last_atr) <= 0:
                    continue
                atr_pct = float(last_atr) / last_close
                if atr_pct < self.min_atr_pct:
                    continue
                tv = 0.0
                if "trading_value" in snap.columns:
                    tv = float(snap.loc[ticker, "trading_value"]) if ticker in snap.index else 0.0
                name = _ticker_name(ticker)
                candidates.append((ticker, name, market, atr_pct, tv))

        candidates.sort(key=lambda x: x[3], reverse=True)

        results: list[tuple[str, dict[str, Any]]] = []
        for ticker, name, market, atr_pct, tv in candidates[: self.n_stocks]:
            cfg: dict[str, Any] = {
                **_SCALPING_CFG_TEMPLATE,
                "name": name,
                "market": market,
                "params": dict(_SCALPING_CFG_TEMPLATE["params"]),
                "_screener": True,
            }
            logger.info(
                "screener: %s %s  ATR %.1f%%  TV KRW%.0fM",
                ticker, name, atr_pct * 100, tv / 1_000_000,
            )
            results.append((ticker, cfg))

        return results
