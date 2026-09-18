"""S&P 500 선물 실시간 방향 모니터.

한국장 개장 중(09:00-15:30 KST)에 E-mini S&P 500 선물(ES=F)을 yfinance로
N분마다 체크한다. 세션 시작 시 기준가를 잡고, 이후 bearish_threshold(기본 -0.5%)
이상 하락하면 BEARISH 플래그를 세워 신규 KRX 진입을 막는다.

yfinance가 없거나 네트워크 오류 시 항상 True(진입 허용)를 반환하므로 봇이
멈추는 일은 없다.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("bot.us_market")


class USMarketMonitor:
    """세션 단위 S&P 500 선물 방향 추적기."""

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("us_market") or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.symbol: str = str(cfg.get("symbol", "ES=F"))
        self.check_interval: float = float(cfg.get("check_interval_seconds", 300))
        self.bearish_threshold: float = float(cfg.get("bearish_threshold", -0.005))
        self._ref_price: float | None = None
        self._last_price: float | None = None
        self._change_pct: float = 0.0
        self._bearish: bool = False
        self._last_check: float = 0.0

    def _fetch_price(self) -> float | None:
        try:
            import yfinance as yf
            price = yf.Ticker(self.symbol).fast_info.last_price
            if price and float(price) > 0:
                return float(price)
        except Exception as exc:
            logger.debug("us_market: yfinance 조회 실패: %s", exc)
        return None

    def check(self) -> bool:
        """True = 진입 허용(중립/상승), False = 진입 차단(하락).

        네트워크 오류 시 True 반환 — 오류로 진입을 막지 않는다.
        """
        if not self.enabled:
            return True

        now = time.monotonic()
        if now - self._last_check < self.check_interval:
            return not self._bearish

        self._last_check = now
        price = self._fetch_price()
        if price is None:
            return True

        if self._ref_price is None:
            self._ref_price = price
            logger.info(
                "us_market: 세션 기준가 %s = %.2f",
                self.symbol, price,
            )

        self._last_price = price
        self._change_pct = (price - self._ref_price) / self._ref_price

        was_bearish = self._bearish
        self._bearish = self._change_pct <= self.bearish_threshold

        if self._bearish and not was_bearish:
            logger.warning(
                "us_market: %s BEARISH 전환 (%.2f%% from %.2f) — 신규 진입 중단",
                self.symbol, self._change_pct * 100, self._ref_price,
            )
        elif not self._bearish and was_bearish:
            logger.info(
                "us_market: %s 회복 (%.2f%%) — 진입 재개",
                self.symbol, self._change_pct * 100,
            )
        else:
            logger.debug(
                "us_market: %s %.2f%% (기준 %.2f, 현재 %.2f)",
                self.symbol, self._change_pct * 100, self._ref_price, price,
            )

        return not self._bearish

    @property
    def status_line(self) -> str:
        if not self.enabled:
            return "us_market: 비활성"
        if self._last_price is None:
            return f"us_market: {self.symbol} 미조회"
        direction = "BEARISH" if self._bearish else "중립/상승"
        return (
            f"us_market: {self.symbol} {direction} "
            f"{self._change_pct:+.2f}% (기준 {self._ref_price:.2f})"
        )
