"""눌림목 반등 (pullback bounce) — 오전장 단타 전략.

상승 흐름(EMA 위) 안에서 일시적으로 눌린(조정) 뒤, 전 봉 고가를 다시 돌파하는
반등 지점에서 진입한다. 채널 돌파(``Scalping``)와 달리 신고가를 쫓지 않고
조정 구간에서 사기 때문에 진입가가 더 낮다.

익절은 3단계다.

1. +arm_pct 미만: 그냥 보유, stop_pct 손절만 작동.
2. +arm_pct~+lock_pct 구간: "무장"됐지만 아직 확정 구간은 아니다 -- 그냥
   기다린다 (여기서 반락해도 exit은 손절가에서만 발생, 즉 최악이라도 원래
   손절폭 이상은 잃지 않는다).
3. +lock_pct 이상: 바닥이 진입가 대비 +lock_pct 아래로는 절대 안 내려가고,
   그 위로는 고점 대비 peak_trail_pct만큼만 밀리면 바로 청산 -- 확정 구간에
   들어온 뒤로는 고점에서 조금만 꺾여도 빠르게 이익을 챙긴다.

peak_trail_pct를 고점 기준으로만 걸면 무장 직후(고점이 arm_pct를 살짝 넘은
수준)에 반락할 때 "익절"이라면서 실제로는 손실로 마감되는 문제가 생긴다 --
그래서 3단계로 나눠 lock_pct 이상에서만 트레일을 건다.
"""

from __future__ import annotations

import pandas as pd

from indicators import (
    bar_strength,
    bollinger_bands,
    ema,
    macd,
    nearest_resistance,
    rolling_max,
    rolling_min,
    rsi,
)
from market.session_rules import parse_clock
from portfolio import Position
from strategies.base import Action, Signal, Strategy
from strategies.scalping import session_vwap


class PullbackBounce(Strategy):
    name = "pullback_bounce"

    def __init__(
        self,
        symbol: str = "",
        timeframe: str = "3Min",
        #: 상승 추세 판정용 EMA 기간.
        trend_ema: int = 20,
        #: 직전 스윙 고점을 찾는 봉 수.
        swing_lookback: int = 10,
        #: 스윙 고점 이후 눌림목 저점을 찾는 최근 봉 수.
        pullback_bars: int = 3,
        #: 스윙 고점 대비 최소 눌림 폭 (0.008 = 0.8%).
        pullback_min_pct: float = 0.008,
        #: 반등 확인 봉의 최소 몸통 강도 (종가가 봉 범위의 몇 %에 위치하는지).
        min_bar_strength: float = 0.3,
        #: 고정 손절 폭.
        stop_pct: float = 0.02,
        #: 정상 stop_pct보다 넓은 손절폭 -- 두 경우에 쓴다 (2026-08-28, 사용자
        #: 요청): (1) 오전 초반(09:00~early_stop_until)은 낙폭 자체가 커서
        #: 정상 stop_pct로는 노이즈에도 자주 걸린다. (2) 추세(EMA) 유지
        #: 중이면 하루 종일 -- 상승 흐름이 진짜 살아있는 동안은 정상 손절폭
        #: 대신 여유를 주고, EMA 아래로 꺾이면 바로 정상 stop_pct로 돌아간다.
        #: 봉 자체의 타임스탬프/종가로 판정하므로 무-룩어헤드 원칙엔 영향 없다.
        early_stop_pct: float = 0.02,
        early_stop_until: str = "09:30",
        #: 이 수익률에 도달하면 "무장" -- 아직 확정 바닥은 아니고 그냥 대기.
        arm_pct: float = 0.02,
        #: 이 수익률부터 진입가 대비 최소 이만큼은 확정 (바닥을 여기 고정).
        lock_pct: float = 0.03,
        #: lock_pct 이상 구간에서 고점 대비 이 폭만큼 밀리면 즉시 청산.
        peak_trail_pct: float = 0.005,
        #: 고점이 이 이상 오른 "진짜 추세" 구간에서는 트레일을 peak_trail_pct
        #: 대신 big_win_trail_pct로 넓혀서, 0.3%대 좁은 트레일 때문에 계속
        #: 오르는 종목을 너무 일찍 털지 않게 한다 (2026-08-27, 사용자 요청).
        big_win_pct: float = 0.04,
        big_win_trail_pct: float = 0.01,
        #: 비용 대비 손절폭 상한 (round_trip_cost_pct / stop_pct 가 이 값을 넘으면
        #: 진입 자체를 막는다 -- 손절폭이 너무 좁아 수수료·세금만 내는 상황 방지).
        max_cost_share: float = 0.35,
        round_trip_cost_pct: float = 0.0038,
        #: RSI 과매수 필터 -- 반등이 이미 다 써버린 상태에서 쫓아 사는 것 방지.
        use_rsi_filter: bool = True,
        rsi_period: int = 14,
        rsi_overbought: float = 75.0,
        #: MACD 히스토그램이 양(+)이거나 개선 중이어야 진입 -- 가격만 오른 게
        #: 아니라 모멘텀 자체가 붙고 있다는 확인.
        use_macd_filter: bool = True,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        #: 최근 고점(저항) 근처면 진입 보류 -- 돌파 여지 없이 바로 막힐 수 있음.
        use_resistance_filter: bool = True,
        resistance_lookback: int = 20,
        resistance_min_room_pct: float = 0.005,
        #: 볼린저밴드(중심선 = 20이평) -- 상단밴드 위로 이미 벗어난 상태에서는
        #: 진입 보류. 자기 자신의 최근 변동성 대비 얼마나 뻗었는지를 본다는
        #: 점에서 RSI와 다르다.
        use_bb_filter: bool = True,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        #: 당일 VWAP(거래량가중평균가) -- 세력의 당일 평균 매수가로 통하는
        #: 지지선. 반등가가 이 아래면 진입 보류 (2026-08-27, 사용자 요청).
        use_vwap_filter: bool = True,
        #: 피보나치 되돌림 -- 눌림 폭이 직전 스윙(swing_low~swing_high) 대비
        #: [fib_min, fib_max] 구간 안에 들어야 진입. 너무 얕은 눌림(추세
        #: 강도 부족 우려)과 너무 깊은 눌림(추세 훼손 우려) 둘 다 거른다
        #: (2026-08-27, 사용자 요청). pullback_min_pct와 달리 스윙 자체의
        #: 크기를 기준으로 삼는다는 점이 다르다.
        use_fib_filter: bool = True,
        fib_min: float = 0.382,
        fib_max: float = 0.5,
        atr_period: int = 14,
        hard_stop_atr_mult: float = 1.0,
        **params,
    ) -> None:
        super().__init__(
            symbol, timeframe,
            atr_period=atr_period, hard_stop_atr_mult=hard_stop_atr_mult,
            **params,
        )
        self.trend_ema = trend_ema
        self.swing_lookback = swing_lookback
        self.pullback_bars = pullback_bars
        self.pullback_min_pct = pullback_min_pct
        self.min_bar_strength = min_bar_strength
        self.stop_pct = stop_pct
        self.early_stop_pct = early_stop_pct
        self.early_stop_until = parse_clock(early_stop_until, "early_stop_until")
        self.arm_pct = arm_pct
        self.lock_pct = lock_pct
        self.peak_trail_pct = peak_trail_pct
        self.big_win_pct = big_win_pct
        self.big_win_trail_pct = big_win_trail_pct
        self.max_cost_share = max_cost_share
        self.round_trip_cost_pct = round_trip_cost_pct
        self.use_rsi_filter = use_rsi_filter
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.use_macd_filter = use_macd_filter
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.use_resistance_filter = use_resistance_filter
        self.resistance_lookback = resistance_lookback
        self.resistance_min_room_pct = resistance_min_room_pct
        self.use_bb_filter = use_bb_filter
        self.bb_period = bb_period
        self.bb_mult = bb_mult
        self.use_vwap_filter = use_vwap_filter
        self.use_fib_filter = use_fib_filter
        self.fib_min = fib_min
        self.fib_max = fib_max

    @property
    def warmup(self) -> int:
        macd_bars = (self.macd_slow + self.macd_signal) if self.use_macd_filter else 0
        rsi_bars = self.rsi_period if self.use_rsi_filter else 0
        bb_bars = self.bb_period if self.use_bb_filter else 0
        return (
            max(self.trend_ema, self.swing_lookback, macd_bars, rsi_bars, bb_bars)
            + self.pullback_bars + 2
        )

    @property
    def window_bars(self) -> int:
        return self.warmup + 30

    def evaluate(self, window: pd.DataFrame, position: Position | None = None) -> Signal:
        if len(window) < self.warmup:
            return self._hold(window, "warming up")

        price = float(window["close"].iloc[-1])
        trend = ema(window["close"], self.trend_ema).iloc[-1]
        if pd.isna(trend):
            return self._hold(window, "trend EMA not established")
        trend = float(trend)

        # 추세(EMA) 유지 중이면 하루 종일 early_stop_pct를 쓴다 (2026-08-28,
        # 사용자 요청): 상승 흐름이 진짜 살아있는 동안은 정상 stop_pct보다
        # 넓게 버텨주고, EMA 아래로 꺾이는 순간 다시 촘촘한 stop_pct로
        # 방어한다. 오전 초반(09:00~early_stop_until)은 추세와 무관하게
        # 낙폭 자체가 커서 항상 넓은 폭을 쓴다 -- 기존 조건은 그대로 두고
        # OR로만 추가했다. window의 마지막 봉 자체 타임스탬프/종가로 판정하므로
        # 무-룩어헤드 원칙에 영향 없음.
        bar_time = pd.Timestamp(window.index[-1]).time()
        trend_intact = price > trend
        effective_stop_pct = (
            self.early_stop_pct
            if (bar_time < self.early_stop_until or trend_intact)
            else self.stop_pct
        )

        # --- manage an open position ---------------------------------------
        if position is not None:
            atr_value = self._atr(window)
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

            if peak_gain >= self.lock_pct:
                # 최소 lock_pct는 확정 -- 바닥이 진입가 아래로 절대 안 내려간다.
                # 그 위로는 고점 대비 트레일 폭만큼만 밀리면 바로 청산 -- 고점이
                # big_win_pct를 넘는 "진짜 추세"에서는 그 폭을 big_win_trail_pct로
                # 넓혀서 좁은 트레일에 너무 일찍 털리지 않게 한다.
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
                # 무장은 됐지만 아직 lock_pct 미달 -- 확정 바닥 없이 그냥 대기
                # (반락해도 위의 stop_pct 손절 이상은 잃지 않는다).
                return self._hold(
                    window,
                    f"무장(+{self.arm_pct:.0%}), {self.lock_pct:.0%} 도달 대기 "
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

        if price <= trend:
            return self._hold(window, f"EMA{self.trend_ema} {trend:,.0f} 아래 -- 상승 추세 아님")

        swing_high = rolling_max(window["high"], self.swing_lookback).shift(1).iloc[-1]
        if pd.isna(swing_high):
            return self._hold(window, "스윙 고점 미형성")
        swing_high = float(swing_high)

        recent = window.iloc[-(self.pullback_bars + 1):-1]
        if recent.empty:
            return self._hold(window, "warming up")
        pullback_low = float(recent["low"].min())
        pullback_depth = (swing_high - pullback_low) / swing_high if swing_high > 0 else 0.0
        if pullback_depth < self.pullback_min_pct:
            return self._hold(window, f"눌림목 아직 (조정폭 {pullback_depth:.2%})")

        # 피보나치 되돌림: pullback_depth(스윙 고점 대비 %)와 달리, 눌림 폭을
        # 직전 스윙 자체의 크기(swing_low~swing_high) 대비 비율로 잰다. 너무
        # 얕으면(fib_min 미만) 추세가 아직 힘이 남아 진짜 눌림이 아닐 수 있고,
        # 너무 깊으면(fib_max 초과) 추세 자체가 훼손됐을 가능성이 크다.
        retracement_pct = None
        if self.use_fib_filter:
            swing_low = rolling_min(window["low"], self.swing_lookback).shift(1).iloc[-1]
            if pd.notna(swing_low):
                swing_range = swing_high - float(swing_low)
                if swing_range > 0:
                    retracement_pct = (swing_high - pullback_low) / swing_range
                    if not (self.fib_min <= retracement_pct <= self.fib_max):
                        return self._hold(
                            window,
                            f"피보나치 되돌림 {retracement_pct:.1%} 범위 밖 "
                            f"(기준 {self.fib_min:.0%}~{self.fib_max:.0%})",
                        )

        prev_high = float(window["high"].iloc[-2])
        if price <= prev_high:
            return self._hold(window, f"반등 미확인 (종가={price:,.0f} <= 전봉고가 {prev_high:,.0f})")

        if self.min_bar_strength > 0:
            strength = float(bar_strength(window).iloc[-1])
            if strength < self.min_bar_strength:
                return self._hold(
                    window,
                    f"반등봉 약함: 범위의 {strength:.0%} (필요 {self.min_bar_strength:.0%})",
                )

        # VWAP: 당일 거래량가중평균가 -- 세력의 당일 평균 매수가로 통하는
        # 지지선. 반등가가 이 아래면 "평균 매수자보다도 낮은 가격"이라 아직
        # 진짜 지지를 못 받은 것으로 본다.
        vwap_now = None
        if self.use_vwap_filter:
            vwap_series = session_vwap(window)
            if pd.notna(vwap_series.iloc[-1]):
                vwap_now = float(vwap_series.iloc[-1])
                if price < vwap_now:
                    return self._hold(
                        window, f"VWAP {vwap_now:,.0f} 아래 -- 세력 평균단가 미회복",
                    )

        # RSI: don't chase a bounce that already spent most of its room.
        rsi_now = None
        if self.use_rsi_filter:
            rsi_now = float(rsi(window["close"], self.rsi_period).iloc[-1])
            if pd.notna(rsi_now) and rsi_now >= self.rsi_overbought:
                return self._hold(
                    window, f"RSI {rsi_now:.0f} 과매수 (기준 {self.rsi_overbought:.0f})",
                )

        # MACD: histogram must confirm momentum is actually building, not just
        # that the last bar ticked up.
        macd_hist = None
        if self.use_macd_filter:
            _, _, hist = macd(window["close"], self.macd_fast, self.macd_slow, self.macd_signal)
            if len(hist) >= 2 and pd.notna(hist.iloc[-1]) and pd.notna(hist.iloc[-2]):
                macd_hist = float(hist.iloc[-1])
                rising = macd_hist > float(hist.iloc[-2])
                if macd_hist <= 0 and not rising:
                    return self._hold(
                        window,
                        f"MACD 모멘텀 부족: 히스토그램 {macd_hist:+.1f} (하락 중)",
                    )

        # Resistance: give the breakout room instead of buying into a ceiling.
        resistance = None
        if self.use_resistance_filter:
            resistance = nearest_resistance(window, self.resistance_lookback, price)
            if resistance is not None:
                room = (resistance - price) / price if price > 0 else 0.0
                if room < self.resistance_min_room_pct:
                    return self._hold(
                        window,
                        f"저항 {resistance:,.0f} 근접 (여유 {room:.2%}, "
                        f"필요 {self.resistance_min_room_pct:.2%})",
                    )

        # Bollinger Bands: block entries already riding above the upper band
        # -- extended relative to the stock's own recent volatility, not just
        # in absolute terms. The middle band is the same 20-period average
        # the trend filter above already uses.
        bb_mid = bb_upper = bb_lower = None
        if self.use_bb_filter:
            mid, upper, lower = bollinger_bands(window["close"], self.bb_period, self.bb_mult)
            if pd.notna(mid.iloc[-1]) and pd.notna(upper.iloc[-1]) and pd.notna(lower.iloc[-1]):
                bb_mid = float(mid.iloc[-1])
                bb_upper = float(upper.iloc[-1])
                bb_lower = float(lower.iloc[-1])
                if price > bb_upper:
                    return self._hold(
                        window,
                        f"볼린저 상단({bb_upper:,.0f}) 위로 이탈 -- 변동성 대비 과도한 확장",
                    )

        atr_value = self._atr(window)
        effective_atr = price * effective_stop_pct
        extra = []
        if vwap_now is not None:
            extra.append(f"VWAP {vwap_now:,.0f}")
        if retracement_pct is not None:
            extra.append(f"피보 {retracement_pct:.1%}")
        if rsi_now is not None:
            extra.append(f"RSI {rsi_now:.0f}")
        if macd_hist is not None:
            extra.append(f"MACD {macd_hist:+.1f}")
        if resistance is not None:
            extra.append(f"저항 {resistance:,.0f}")
        if bb_mid is not None:
            extra.append(f"BB중심 {bb_mid:,.0f}/상단 {bb_upper:,.0f}")
        extra_note = f" ({', '.join(extra)})" if extra else ""
        return self._signal(
            window, Action.ENTER_LONG,
            f"눌림목 반등: 고점 {swing_high:,.0f} 대비 {pullback_depth:.2%} 조정 후 "
            f"{prev_high:,.0f} 재돌파 [손절 {effective_stop_pct:.1%}, 무장 {self.arm_pct:.0%}, "
            f"확정 {self.lock_pct:.0%}, 트레일 {self.peak_trail_pct:.1%}]{extra_note}",
            effective_atr if effective_atr > 0 else atr_value,
            meta={
                "swing_high": swing_high,
                "pullback_low": pullback_low,
                "pullback_depth": pullback_depth,
                "retracement_pct": retracement_pct,
                "vwap": vwap_now,
            },
        )

    def update_trailing_stop(self, window: pd.DataFrame, position: Position) -> float | None:
        # 트레일은 evaluate()의 EXIT 시그널로 직접 처리한다 (peak_gain 기준).
        # ATR 트레일을 쓰지 않으므로 stop_price(고정 손절)를 그대로 둔다.
        return None
