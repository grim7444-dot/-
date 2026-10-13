"""일봉 추세 확인 -- 다중 시간프레임(상위 프레임) 필터.

3분봉 하나만 보고 진입을 결정하면, 상위 추세와 반대 방향인 반등을 "눌림목"
이라고 착각해서 살 위험이 있다. 세션당 한 번 각 종목의 최근 일봉으로
EMA(기본 20일) 추세를 계산해 캐시해 두고, 일봉 자체가 하락 추세인 종목은
3분봉에서 아무리 좋은 신호가 나와도 진입을 막는다.

투자자 흐름(investor_flow.py)과 같은 패턴: 세션 시작 시 한 번 스캔하고,
데이터가 없으면 막지 않는다(advisory 기본값과 별개로, 여기서는 아예 데이터
부재 시 통과시켜 이 필터 하나 때문에 봇이 멈추는 일이 없게 한다).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from indicators import ema

logger = logging.getLogger("bot.daily_trend")


class DailyTrendScanner:
    """세션 시작 시 한 번, 종목별 일봉 EMA 추세 방향을 캐시한다."""

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("daily_trend") or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.ema_period: int = int(cfg.get("ema_period", 20))
        self._bias: dict[str, bool] = {}  # code -> True(상승) / False(하락)
        self._scan_date: date | None = None

    def scan(self, market_data: Any, universe: dict[str, Any]) -> None:
        today = date.today()
        if self._scan_date == today:
            return
        self._scan_date = today
        self._bias.clear()
        if not self.enabled:
            return

        for code, asset_cfg in universe.items():
            try:
                barset = market_data.get_bars(
                    code=code,
                    timeframe="1Day",
                    lookback_bars=self.ema_period + 30,
                    market=asset_cfg.get("market", "KOSPI"),
                )
                frame = getattr(barset, "bars", barset)
                if frame is None or frame.empty or len(frame) < self.ema_period + 2:
                    continue
                trend = ema(frame["close"], self.ema_period)
                last_trend = float(trend.iloc[-1])
                if last_trend != last_trend:  # NaN
                    continue
                last_close = float(frame["close"].iloc[-1])
                is_up = last_close > last_trend
                self._bias[code] = is_up
                logger.info(
                    "daily_trend %s: 종가 %.0f %s EMA%d %.0f -> %s",
                    code, last_close, ">" if is_up else "<=",
                    self.ema_period, last_trend, "상승" if is_up else "하락",
                )
            except Exception as exc:
                logger.debug("daily_trend %s: 조회 실패: %s", code, exc)

    def allows_long(self, code: str) -> bool:
        """일봉 데이터가 없으면 막지 않는다 -- 이 필터 하나로 봇이 멈추지 않게."""
        return self._bias.get(code, True)
