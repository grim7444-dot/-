"""거시 지표 게이지 -- 미국증시 마감강세, WTI유, 미국10년물 금리, 비트코인, 달러지수.

사용자 확인: 다섯 지표 모두 "상승 = 강세(공격 모드)"로 취급한다. 실제 매크로
상관관계는 더 복잡하다 -- 예를 들어 달러지수 강세는 보통 신흥국 증시엔
악재로 통하지만, 사용자가 명시적으로 "다섯 다 오르면 강세"로 정의했으므로
그 정의를 그대로 구현한다.

하루 한 번, KRX 세션 시작 시(정확히는 세션 중 첫 스캔 시) 전일 종가 대비
등락을 확인해 상승 지표 개수를 센다. bullish_threshold개 이상 상승이면
"공격 모드"로 판단하고, main.py의 TradingEngine이 이 상태를 읽어 진입 조건
완화 / 동시 보유 종목 수 확대 / 종목당 리스크 확대에 쓴다 (실제 적용은
main.py 쪽 책임 -- 이 모듈은 판정만 한다).

yfinance 의존 -- 없거나 네트워크 오류면 해당 지표를 집계에서 빼고, 하나도
못 가져오면 게이지 전체가 중립(기본 모드)으로 남는다. 이 게이지 하나 때문에
봇이 멈추거나 막히는 일은 없다.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

logger = logging.getLogger("bot.macro")

#: (표시이름, yfinance 티커). 다섯 다 "상승 = 강세"로 취급한다.
_INDICATORS: tuple[tuple[str, str], ...] = (
    ("미국증시(S&P500)", "^GSPC"),
    ("WTI유", "CL=F"),
    ("미국10년물금리", "^TNX"),
    ("비트코인", "BTC-USD"),
    ("달러지수", "DX-Y.NYB"),
)


class MacroGauge:
    """하루 한 번 5개 거시 지표를 조회해 상승 개수를 세는 세션 단위 게이지."""

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("macro_gauge") or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        #: 5개 중 이 개수 이상 상승이면 "강세장(공격 모드)"으로 판단.
        self.bullish_threshold: int = int(cfg.get("bullish_threshold", 4))
        #: 강세일 때 진입 조건(거래량배수/봉강도/눌림폭)을 완화하는 비율.
        self.entry_relax_pct: float = float(cfg.get("entry_relax_pct", 0.15))
        #: 강세일 때 동시 보유 가능 종목 수 가산치.
        self.extra_positions: int = int(cfg.get("extra_positions", 1))
        #: 강세일 때 종목당 리스크(risk_pct) 배수.
        self.risk_pct_boost: float = float(cfg.get("risk_pct_boost", 1.2))

        self._scan_date: date | None = None
        self._bullish_count: int = 0
        self._details: list[str] = []

    @property
    def is_bullish(self) -> bool:
        return self.enabled and self._bullish_count >= self.bullish_threshold

    @property
    def bullish_count(self) -> int:
        return self._bullish_count

    @property
    def status_line(self) -> str:
        if not self.enabled:
            return "macro_gauge: 비활성"
        if not self._details:
            return "macro_gauge: 미조회"
        mode = "공격 모드" if self.is_bullish else "기본 모드"
        return f"macro_gauge: {self._bullish_count}/5 강세 [{', '.join(self._details)}] -- {mode}"

    def scan(self) -> None:
        """오늘 아직 안 했으면 5개 지표를 조회하고, 했으면 그냥 리턴 (세션당 1회)."""
        today = date.today()
        if self._scan_date == today:
            return
        self._scan_date = today
        self._bullish_count = 0
        self._details = []
        if not self.enabled:
            return

        try:
            import yfinance as yf
        except ImportError:
            logger.debug("macro_gauge: yfinance 미설치 -- 게이지 비활성 취급")
            return

        for label, ticker in _INDICATORS:
            try:
                hist = yf.Ticker(ticker).history(period="5d")
                closes = hist["Close"].dropna() if hist is not None else None
                if closes is None or len(closes) < 2:
                    logger.debug("macro_gauge: %s(%s) 데이터 부족", label, ticker)
                    continue
                prev, last = float(closes.iloc[-2]), float(closes.iloc[-1])
                if prev <= 0:
                    continue
                chg = (last - prev) / prev
                bullish = chg > 0
                if bullish:
                    self._bullish_count += 1
                self._details.append(f"{label}{chg:+.2%}")
            except Exception as exc:
                logger.debug("macro_gauge: %s(%s) 조회 실패: %s", label, ticker, exc)

        logger.info("%s", self.status_line)
