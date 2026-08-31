"""오프닝레인지 브레이크아웃 (ORB) — 오전장 단타 전략.

장 시작 range_minutes(기본 15분) 동안 형성된 고저를 "오늘의 레인지"로 고정한
뒤, 그 위로 거래량을 동반해 종가가 돌파하면 진입한다. 기관 데스크가 실제로
쓰는 정석 패턴: 첫 몇 분의 반응성 매매 노이즈를 걸러내고, 그 구간에서 형성된
매물대를 실제로 뚫을 만한 힘이 있는지를 본다.

진입 조건:

1. 종가가 레인지 고점 위로 마감 (거래량 동반)
2. trend_ema 위 -- 상위 추세와 같은 방향인지 확인
3. 반등봉 강도(bar_strength) -- 종가가 봉 상단 근처에서 마감했는지
4. (기본 비활성) 세션 VWAP 위 -- 레인지는 뚫었지만 VWAP 아래면 기관 매수
   없이 개인들만 반응한 "가짜 돌파(buyer's trap)"일 가능성이 크다는
   취지였지만, 강한 진짜 돌파에서도 레인지는 이미 뚫었는데 당일 누적
   VWAP은 아직 못 넘은 구간이 한동안 이어지는 경우가 많아 진입을 필요
   이상으로 늦춘다는 사용자 피드백 (2026-08-31) -- use_vwap_filter로
   다시 켤 수 있다.

익절/손절은 눌림목 반등(PullbackBounce)과 동일한 3단계 트레일을 그대로
쓴다 -- 이미 검증된 로직을 재사용해 익절 방식의 일관성을 유지한다.
"""

from __future__ import annotations

import pandas as pd

from indicators import bar_strength, ema, rolling_mean_volume
from market.session_rules import parse_clock
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
        #: 세션 VWAP 위 확인 -- 기본 비활성 (2026-08-31, 사용자 요청: "돌파매매도
        #: 느리고" -- 레인지는 이미 뚫었는데 당일 누적 VWAP은 아직 못 넘은
        #: 구간이 한동안 이어지는 경우가 많아 진입을 필요 이상으로 늦춘다는
        #: 피드백. 대신 개인 매수만으로 반짝 뚫었다 되돌리는 가짜 돌파에 물릴
        #: 위험은 커진다 -- 다시 켜려면 True로.
        use_vwap_filter: bool = False,
        #: 고정 손절 폭.
        stop_pct: float = 0.017,
        #: 정상 stop_pct보다 넓은 손절폭 -- 두 경우에 쓴다 (2026-08-28, 사용자
        #: 요청): (1) 오전 초반(09:00~early_stop_until)은 낙폭 자체가 커서
        #: 정상 stop_pct로는 노이즈에도 자주 걸린다. (2) 추세(EMA) 유지
        #: 중이면 하루 종일 -- 상승 흐름이 진짜 살아있는 동안은 정상 손절폭
        #: 대신 여유를 주고, EMA 아래로 꺾이면 바로 정상 stop_pct로 돌아간다.
        #: 봉 자체의 타임스탬프/종가로 판정하므로(오프닝 레인지와 동일한
        #: 방식) 무-룩어헤드 원칙은 그대로 유지된다.
        early_stop_pct: float = 0.02,
        early_stop_until: str = "09:30",
        #: 오후장(midday_window_start~midday_window_end) 시간대에 새로
        #: 진입하는 포지션은 손절/확정 폭을 대칭 1.5%/1.5%로 고정 (2026-08-28,
        #: 사용자 요청: "너무 거래가 없네 -- 익절1.5% 손절1.5%로 가자
        #: 11시부터3시까지"). 회전을 빠르게 해서 체결 빈도를 늘리려는
        #: 목적이라 early_stop_pct/트레일 확대보다 우선한다. 진입 시점(그
        #: 봉의 타임스탬프)에 한 번 정해지면 포지션이 살아있는 동안
        #: entry_time 기준으로 계속 유지된다 -- 15시를 넘겨 보유 중이어도
        #: 시계가 바뀌었다고 도중에 다른 티어로 전환되지 않는다.
        midday_stop_pct: float = 0.015,
        midday_lock_pct: float = 0.015,
        midday_window_start: str = "11:00",
        midday_window_end: str = "15:00",
        #: PullbackBounce와 동일한 3단계 익절 (무장/확정/고점트레일).
        arm_pct: float = 0.012,
        lock_pct: float = 0.018,
        peak_trail_pct: float = 0.005,
        #: 고점이 이 이상 오른 "진짜 추세" 구간에서는 트레일을 peak_trail_pct
        #: 대신 big_win_trail_pct로 넓혀서, 좁은 트레일 때문에 계속 오르는
        #: 종목을 너무 일찍 털지 않게 한다 (2026-08-27, 사용자 요청). 고정
        #: 바닥(entry 기준)을 시도했다가(2026-08-31 초안) 사용자가 "안되,
        #: 고점 대비 하락"으로 정정 -- 그대로 고점 대비 트레일로 되돌리고
        #: 문턱은 4%, 폭은 2%로 조정했다.
        big_win_pct: float = 0.04,
        big_win_trail_pct: float = 0.02,
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
        self.use_vwap_filter = use_vwap_filter
        self.stop_pct = stop_pct
        self.early_stop_pct = early_stop_pct
        self.early_stop_until = parse_clock(early_stop_until, "early_stop_until")
        self.midday_stop_pct = midday_stop_pct
        self.midday_lock_pct = midday_lock_pct
        self.midday_window_start = parse_clock(midday_window_start, "midday_window_start")
        self.midday_window_end = parse_clock(midday_window_end, "midday_window_end")
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

        trend_val = ema(window["close"], self.trend_ema).iloc[-1]
        trend_now = float(trend_val) if pd.notna(trend_val) else None
        # 추세(EMA) 유지 중이면 하루 종일 early_stop_pct를 쓴다 (2026-08-28,
        # 사용자 요청): 상승 흐름이 진짜 살아있는 동안은 정상 stop_pct보다
        # 넓게 버텨주고, EMA 아래로 꺾이는 순간 다시 촘촘한 stop_pct로
        # 방어한다. 오전 초반(09:00~early_stop_until)은 추세와 무관하게
        # 낙폭 자체가 커서 항상 넓은 폭을 쓴다 -- 기존 조건은 그대로 두고
        # OR로만 추가했다. window의 마지막 봉 자체 타임스탬프/종가로 판정하므로
        # 무-룩어헤드 원칙에 영향 없음.
        bar_time = pd.Timestamp(window.index[-1]).time()
        trend_intact = trend_now is not None and price > trend_now
        # 오후장 진입은 entry_time(포지션이 없으면 이번 봉 자체)이 기준 --
        # 한 번 정해지면 그 포지션이 살아있는 동안 바뀌지 않는다.
        reference_time = (
            (position.entry_time_of_day() or bar_time) if position is not None else bar_time
        )
        is_midday = self.midday_window_start <= reference_time < self.midday_window_end
        effective_stop_pct = (
            self.midday_stop_pct if is_midday
            else self.early_stop_pct if (bar_time < self.early_stop_until or trend_intact)
            else self.stop_pct
        )
        effective_lock_pct = self.midday_lock_pct if is_midday else self.lock_pct

        # --- manage an open position -- identical tiering to PullbackBounce ---
        if position is not None:
            entry = position.entry_price
            gain = (price - entry) / entry if entry else 0.0
            peak = position.highest_price or price
            peak_gain = (peak - entry) / entry if entry else 0.0

            stop_price = entry * (1 - effective_stop_pct)
            if price <= stop_price:
                return self._signal(
                    window, Action.EXIT,
                    f"{effective_stop_pct:.1%} 손절 ({gain:+.2%})", atr_value,
                )

            if peak_gain >= effective_lock_pct:
                trail_pct = (
                    self.big_win_trail_pct if peak_gain >= self.big_win_pct
                    else self.peak_trail_pct
                )
                floor = max(
                    entry * (1 + effective_lock_pct),
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
                    f"무장(+{self.arm_pct:.0%}), {effective_lock_pct:.0%} 도달 대기 "
                    f"(현재 {gain:+.2%})",
                )
            return self._hold(
                window, f"미무장, 보유 중 ({gain:+.2%}, {self.arm_pct:.0%} 도달 시 무장)",
            )

        # --- entries ----------------------------------------------------
        if effective_stop_pct > 0:
            cost_share = self.round_trip_cost_pct / effective_stop_pct
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

        if trend_now is None:
            return self._hold(window, "trend EMA not established")
        if price <= trend_now:
            return self._hold(window, f"EMA{self.trend_ema} {trend_now:,.0f} 아래 -- 상승 추세 아님")

        vwap_note = ""
        if self.use_vwap_filter:
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
            vwap_note = f", VWAP {vwap_now:,.0f} 위"

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

        effective_atr = price * effective_stop_pct
        return self._signal(
            window, Action.ENTER_LONG,
            f"ORB: 오프닝 레인지({self.range_minutes}분) 고점 {range_high:,.0f} 돌파 "
            f"{vol_ratio:.2f}x 거래량{vwap_note}, EMA{self.trend_ema} 위 "
            f"[손절 {effective_stop_pct:.1%}, 무장 {self.arm_pct:.0%}, 확정 {effective_lock_pct:.0%}]",
            effective_atr if effective_atr > 0 else atr_value,
            meta={"range_high": range_high, "range_low": range_low, "volume_ratio": vol_ratio},
        )

    def update_trailing_stop(self, window: pd.DataFrame, position: Position) -> float | None:
        return None
