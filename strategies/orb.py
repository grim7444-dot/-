"""오프닝레인지 브레이크아웃 (ORB) — 오전장 단타 전략.

장 시작 range_minutes(기본 15분) 동안 형성된 고저를 "오늘의 레인지"로 고정한
뒤, 그 위로 거래량을 동반해 종가가 돌파하면 진입한다. 기관 데스크가 실제로
쓰는 정석 패턴: 첫 몇 분의 반응성 매매 노이즈를 걸러내고, 그 구간에서 형성된
매물대를 실제로 뚫을 만한 힘이 있는지를 본다.

진입 조건은 4중이다 -- 레인지 돌파 하나만으로는 부족하다:

1. 종가가 레인지 고점 위로 마감 (거래량 동반)
2. 세션 VWAP 위 -- 레인지는 뚫었지만 VWAP 아래면 기관 매수 없이 개인들만
   반응한 "가짜 돌파(buyer's trap)"일 가능성이 크다
3. trend_ema 위 -- 상위 추세와 같은 방향인지 확인
4. 반등봉 강도(bar_strength) -- 종가가 봉 상단 근처에서 마감했는지

익절/손절은 눌림목 반등(PullbackBounce)과 동일한 3단계 트레일을 그대로
쓴다 -- 이미 검증된 로직을 재사용해 익절 방식의 일관성을 유지한다.
"""

from __future__ import annotations

import pandas as pd

from indicators import bar_strength, ema, rolling_mean_volume
from portfolio import Position
from strategies.base import Action, Signal, Strategy
from strategies.scalping import session_vwap


def _opening_range(
    window: pd.DataFrame, range_minutes: int, session_open_hour: int, session_open_minute: int
) -> tuple[float, float, bool] | None:
    """(range_high, range_low, range_complete) for *today's* session, or None.

    ``range_complete`` is False while the opening window itself is still
    forming -- entries must wait for it, not trade inside it.
    """
    idx = pd.DatetimeIndex(window.index)
    today = idx[-1].normalize()
    today_bars = window[idx.normalize() == today]
    if today_bars.empty:
        return None
    session_open = today + pd.Timedelta(hours=session_open_hour, minutes=session_open_minute)
    range_end = session_open + pd.Timedelta(minutes=range_minutes)
    range_bars = today_bars[
        (today_bars.index >= session_open) & (today_bars.index < range_end)
    ]
    if range_bars.empty:
        return None
    complete = today_bars.index[-1] >= range_end
    return float(range_bars["high"].max()), float(range_bars["low"].min()), complete


class ORB(Strategy):
    name = "orb"

    def __init__(
        self,
        symbol: str = "",
        timeframe: str = "3Min",
        #: 오프닝 레인지 길이(분). 연구 결과 15분이 노이즈-기회 균형의
        #: "스위트 스팟"으로 꼽힌다.
        range_minutes: int = 15,
        session_open_hour: int = 9,
        session_open_minute: int = 0,
        #: 돌파 확인용 평균거래량 lookback 봉 수.
        volume_lookback: int = 10,
        volume_mult: float = 1.5,
        #: 상위 추세 판정용 EMA.
        trend_ema: int = 21,
        min_bar_strength: float = 0.5,
        #: 고정 손절 폭.
        stop_pct: float = 0.017,
        #: PullbackBounce와 동일한 3단계 익절 (무장/확정/고점트레일).
        arm_pct: float = 0.012,
        lock_pct: float = 0.018,
        peak_trail_pct: float = 0.005,
        #: 고점이 이 이상 오른 "진짜 추세" 구간에서는 트레일을 peak_trail_pct
        #: 대신 big_win_trail_pct로 넓혀서, 0.3%대 좁은 트레일 때문에 계속
        #: 오르는 종목을 너무 일찍 털지 않게 한다 (2026-08-27, 사용자 요청).
        big_win_pct: float = 0.04,
        big_win_trail_pct: float = 0.01,
        max_cost_share: float = 0.35,
        round_trip_cost_pct: float = 0.0038,
        atr_period: int = 14,
        hard_stop_atr_mult: float = 1.0,
        **params,
    ) -> None:
        super().__init__(
            symbol, timeframe,
            atr_period=atr_period, hard_stop_atr_mult=hard_stop_atr_mult,
            **params,
        )
        self.range_minutes = range_minutes
        self.session_open_hour = session_open_hour
        self.session_open_minute = session_open_minute
        self.volume_lookback = volume_lookback
        self.volume_mult = volume_mult
        self.trend_ema = trend_ema
        self.min_bar_strength = min_bar_strength
        self.stop_pct = stop_pct
        self.arm_pct = arm_pct
        self.lock_pct = lock_pct
        self.peak_trail_pct = peak_trail_pct
        self.big_win_pct = big_win_pct
        self.big_win_trail_pct = big_win_trail_pct
        self.max_cost_share = max_cost_share
        self.round_trip_cost_pct = round_trip_cost_pct

    _SESSION_BARS = {
        "1Min": 390, "3Min": 130, "5Min": 78, "10Min": 39,
        "15Min": 26, "30Min": 13, "60Min": 7,
    }

    @property
    def warmup(self) -> int:
        return max(self.trend_ema, self.volume_lookback) + 2

    @property
    def window_bars(self) -> int:
        per_session = self._SESSION_BARS.get(self.timeframe, 130)
        return max(self.warmup, per_session) + 2 * per_session

    def evaluate(self, window: pd.DataFrame, position: Position | None = None) -> Signal:
        if len(window) < self.warmup:
            return self._hold(window, "warming up")

        price = float(window["close"].iloc[-1])
        atr_value = self._atr(window)

        # --- manage an open position -- identical tiering to PullbackBounce ---
        if position is not None:
            entry = position.entry_price
            gain = (price - entry) / entry if entry else 0.0
            peak = position.highest_price or price
            peak_gain = (peak - entry) / entry if entry else 0.0

            stop_price = entry * (1 - self.stop_pct)
            if price <= stop_price:
                return self._signal(
                    window, Action.EXIT,
                    f"{self.stop_pct:.0%} 손절 ({gain:+.2%})", atr_value,
                )

            if peak_gain >= self.lock_pct:
                trail_pct = (
                    self.big_win_trail_pct if peak_gain >= self.big_win_pct
                    else self.peak_trail_pct
                )
                floor = max(
                    entry * (1 + self.lock_pct),
                    peak * (1.0 - trail_pct),
                )
                if price <= floor:
                    return self._signal(
                        window, Action.EXIT,
                        f"고점 +{peak_gain:.2%}에서 반락 -- {gain:+.2%} 확정 익절",
                        atr_value,
                    )
                return self._hold(
                    window, f"익절 대기, 고점 +{peak_gain:.2%} (현재 {gain:+.2%})",
                )

            if peak_gain >= self.arm_pct:
                return self._hold(
                    window,
                    f"무장(+{self.arm_pct:.0%}), {self.lock_pct:.0%} 도달 대기 "
                    f"(현재 {gain:+.2%})",
                )
            return self._hold(
                window, f"미무장, 보유 중 ({gain:+.2%}, {self.arm_pct:.0%} 도달 시 무장)",
            )

        # --- entries ----------------------------------------------------
        if self.stop_pct > 0:
            cost_share = self.round_trip_cost_pct / self.stop_pct
            if cost_share > self.max_cost_share:
                return self._hold(
                    window,
                    f"손절폭 대비 비용 과다: {cost_share:.0%} "
                    f"(상한 {self.max_cost_share:.0%})",
                )

        rng = _opening_range(
            window, self.range_minutes, self.session_open_hour, self.session_open_minute
        )
        if rng is None:
            return self._hold(window, "오프닝 레인지 데이터 없음")
        range_high, range_low, complete = rng
        if not complete:
            return self._hold(
                window, f"오프닝 레인지 형성 중 (첫 {self.range_minutes}분)",
            )

        trend = ema(window["close"], self.trend_ema).iloc[-1]
        if pd.isna(trend):
            return self._hold(window, "trend EMA not established")
        trend = float(trend)
        if price <= trend:
            return self._hold(window, f"EMA{self.trend_ema} {trend:,.0f} 아래 -- 상승 추세 아님")

        vwap = session_vwap(window)
        vwap_now = vwap.iloc[-1]
        if pd.isna(vwap_now):
            return self._hold(window, "VWAP not established")
        vwap_now = float(vwap_now)
        if price <= vwap_now:
            return self._hold(
                window, f"VWAP {vwap_now:,.0f} 아래 -- 레인지는 뚫었어도 매수세 없음"
                if price > range_high else f"레인지 고점 {range_high:,.0f} 미돌파",
            )

        if price <= range_high:
            return self._hold(window, f"레인지 고점 {range_high:,.0f} 미돌파 (레인지 저점 {range_low:,.0f})")

        avg_volume = rolling_mean_volume(window, self.volume_lookback).shift(1).iloc[-1]
        volume = float(window["volume"].iloc[-1])
        vol_ratio = (volume / avg_volume) if avg_volume and avg_volume > 0 else 0.0
        if not (avg_volume and avg_volume > 0 and vol_ratio >= self.volume_mult):
            return self._hold(
                window,
                f"돌파 거부: 거래량 {vol_ratio:.2f}x < {self.volume_mult}x",
            )

        if self.min_bar_strength > 0:
            strength = float(bar_strength(window).iloc[-1])
            if strength < self.min_bar_strength:
                return self._hold(
                    window,
                    f"돌파봉 약함: 범위의 {strength:.0%} (필요 {self.min_bar_strength:.0%})",
                )

        effective_atr = price * self.stop_pct
        return self._signal(
            window, Action.ENTER_LONG,
            f"ORB: 오프닝 레인지({self.range_minutes}분) 고점 {range_high:,.0f} 돌파 "
            f"{vol_ratio:.2f}x 거래량, VWAP {vwap_now:,.0f} 위, EMA{self.trend_ema} 위 "
            f"[손절 {self.stop_pct:.0%}, 무장 {self.arm_pct:.0%}, 확정 {self.lock_pct:.0%}]",
            effective_atr if effective_atr > 0 else atr_value,
            meta={"range_high": range_high, "range_low": range_low, "volume_ratio": vol_ratio},
        )

    def update_trailing_stop(self, window: pd.DataFrame, position: Position) -> float | None:
        return None
