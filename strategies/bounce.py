"""Buy a technical bounce off a sharp intraday crash (반등매매).

From a shared blog article's own spec (2026-09-16, user request): a stock
down ``crash_pct`` or more from the previous close is a candidate, but only
once it actually shows signs of turning -- price already ``bounce_confirm_pct``
off its own post-crash low, on ``volume_mult`` times normal volume, closing
strong (``min_bar_strength``) and above a short ``support_ema``. Skipping any
one of those is "catching a falling knife": the article's own tip is
"반등 없는 종목 많음" (many crashed stocks never bounce at all) -- the whole
point of these filters is separating an actual reversal from one more leg
down.

The exit is deliberately simple and fixed, matching the article's own "반등
3~5% 수익 시 매도": a flat take_profit_pct target, no trailing, no tiered
lock. This is a fast, disciplined scalp on a stock that just proved it can
move hard in either direction, not a trend-following hold -- the hard stop
(set from ``stop_pct``, enforced by the engine's own per-cycle check, not by
anything in this file) is what a falling knife actually needs, not room to
run.
"""

from __future__ import annotations

import pandas as pd

from indicators import bar_strength, ema, rolling_mean_volume
from portfolio import Position
from strategies.base import Action, Signal, Strategy


def _previous_close(window: pd.DataFrame) -> float | None:
    """The last close from the most recent earlier session in *window*.

    None when the window does not yet span more than one session (e.g. a
    fresh morning before enough history has accumulated) -- there is
    nothing to measure a crash against yet.
    """
    days = pd.Index(window.index).normalize()
    today = days[-1]
    prior = window.loc[days < today]
    if prior.empty:
        return None
    return float(prior["close"].iloc[-1])


class BounceReversal(Strategy):
    name = "bounce"

    def __init__(
        self,
        symbol: str = "",
        timeframe: str = "5Min",
        #: Must be down at least this much from the previous session's close.
        crash_pct: float = 0.07,
        #: Window used to find the post-crash low the bounce is measured from.
        low_lookback: int = 20,
        #: Price must already be this far off that low -- confirms a bounce
        #: is underway rather than catching a still-falling stock.
        bounce_confirm_pct: float = 0.015,
        volume_period: int = 20,
        volume_mult: float = 1.5,
        #: Price must sit above this EMA (0 disables) -- "이평선 위에서 반등
        #: 나와야 안정적".
        support_ema: int = 5,
        #: Bar must close in at least this much of its own range (0=low,
        #: 1=high; 0 disables) -- buyers, not sellers, controlled the bounce
        #: bar itself.
        min_bar_strength: float = 0.5,
        stop_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        atr_period: int = 14,
        hard_stop_atr_mult: float = 1.0,
        allow_short: bool = False,
        **params,
    ) -> None:
        super().__init__(
            symbol, timeframe,
            atr_period=atr_period, hard_stop_atr_mult=hard_stop_atr_mult,
            **params,
        )
        self.crash_pct = crash_pct
        self.low_lookback = low_lookback
        self.bounce_confirm_pct = bounce_confirm_pct
        self.volume_period = volume_period
        self.volume_mult = volume_mult
        self.support_ema = support_ema
        self.min_bar_strength = min_bar_strength
        self.stop_pct = stop_pct
        self.take_profit_pct = take_profit_pct
        self.allow_short = allow_short

    _SESSION_BARS = {
        "1Min": 390, "3Min": 130, "5Min": 78, "10Min": 39,
        "15Min": 26, "30Min": 13, "60Min": 7,
    }

    @property
    def warmup(self) -> int:
        ema_bars = self.support_ema if self.support_ema > 0 else 0
        return max(self.low_lookback, self.volume_period, self.atr_period, ema_bars) + 2

    @property
    def window_bars(self) -> int:
        """Two sessions of margin: one for the previous close, one so the
        current session is complete even starting mid-day (same convention
        as Scalping.window_bars)."""
        per_session = self._SESSION_BARS.get(self.timeframe, 78)
        return self.warmup + 2 * per_session

    def evaluate(
        self, window: pd.DataFrame, position: Position | None = None,
        market_down: bool = False,
    ) -> Signal:
        if len(window) < self.warmup:
            return self._hold(window, "warming up")

        price = float(window["close"].iloc[-1])
        atr_value = self._atr(window)

        # --- manage an open position -- fixed target, no trail; the hard
        # stop from stop_distance is enforced by the engine itself. ---
        if position is not None:
            entry = position.entry_price
            gain = (price - entry) / entry if entry else 0.0
            take_profit = position.take_profit
            if take_profit and price >= take_profit:
                return self._signal(
                    window, Action.EXIT,
                    f"반등 목표 +{self.take_profit_pct:.0%} 도달 ({gain:+.2%})",
                    atr_value,
                )
            return self._hold(
                window, f"반등 대기 중 ({gain:+.2%}, 목표 +{self.take_profit_pct:.0%})"
            )

        # --- entries ---------------------------------------------------
        prev_close = _previous_close(window)
        if prev_close is None or prev_close <= 0:
            return self._hold(window, "전일 종가 없음 -- 급락 여부 판단 불가")
        drop_pct = (prev_close - price) / prev_close
        if drop_pct < self.crash_pct:
            return self._hold(
                window,
                f"전일 대비 {drop_pct:+.1%} -- 급락 기준 {self.crash_pct:.0%} 미달",
            )

        recent_low = float(window["low"].tail(self.low_lookback).min())
        if recent_low <= 0:
            return self._hold(window, "최근 저점 없음")
        bounce_pct = (price - recent_low) / recent_low
        if bounce_pct < self.bounce_confirm_pct:
            return self._hold(
                window,
                f"저점({recent_low:,.0f}) 대비 +{bounce_pct:.1%} -- "
                f"반등 확인 기준 {self.bounce_confirm_pct:.1%} 미달",
            )

        if self.support_ema > 0:
            support = ema(window["close"], self.support_ema).iloc[-1]
            if pd.isna(support) or price <= float(support):
                return self._hold(
                    window, f"{self.support_ema}봉 이평선 아래 -- 반등 불안정"
                )

        avg_volume = rolling_mean_volume(window, self.volume_period).shift(1).iloc[-1]
        volume = float(window["volume"].iloc[-1])
        if pd.isna(avg_volume) or float(avg_volume) <= 0:
            return self._hold(window, "volume average not established")
        vol_ratio = volume / float(avg_volume)
        if vol_ratio < self.volume_mult:
            return self._hold(
                window,
                f"반등 거래량 {vol_ratio:.2f}x < {self.volume_mult}x -- 거래량 부족",
            )

        if self.min_bar_strength > 0:
            strength = float(bar_strength(window).iloc[-1])
            if strength < self.min_bar_strength:
                return self._hold(
                    window,
                    f"반등봉 약세 마감 {strength:.0%} of range "
                    f"(기준 {self.min_bar_strength:.0%} -- 매도세 우위)",
                )

        take_profit = price * (1 + self.take_profit_pct) if self.take_profit_pct > 0 else None
        effective_atr = price * self.stop_pct if self.stop_pct > 0 else atr_value
        return self._signal(
            window,
            Action.ENTER_LONG,
            f"급락 반등: 전일대비 {drop_pct:.1%}, 저점대비 +{bounce_pct:.1%}, "
            f"{vol_ratio:.2f}x 거래량 [손절 {self.stop_pct:.0%}, 목표 +{self.take_profit_pct:.0%}]",
            effective_atr,
            take_profit=take_profit,
            meta={
                "drop_pct": drop_pct,
                "bounce_pct": bounce_pct,
                "volume_ratio": vol_ratio,
            },
        )
