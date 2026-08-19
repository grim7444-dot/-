"""Buy a strong close, sell the next morning (종가매매).

The trade is a bet on one specific thing: a stock that finishes the session
near its high, on heavier volume than usual, tends to open the next session
higher. Entry is in the last minutes of continuous trading; the exit is a
planned one the following morning, and both times come from ``SessionRules``
rather than from anything here -- a daily bar is stamped midnight, so a
strategy reading daily bars cannot tell what time it is.

Three conditions, all measured on today's own bar as it stands:

* the price sits in the top ``close_strength`` of today's range -- a stock
  closing on its low is the opposite trade;
* it is above its ``trend_ema``-day EMA, so the day's strength is happening
  inside an uptrend rather than as a bounce in a decline;
* today's volume is at least ``volume_mult`` times the recent average, which
  is what separates a strong close from a quiet drift upward.

**The overnight gap is not covered by any stop.** Kiwoom holds no resting stop
order, so the hard stop exists only while the bot process is running -- and
between the close and the next open, it is not. A stock that gaps through the
stop opens the position at whatever the auction decides. That risk is the
price of the trade and cannot be engineered away here; it is why
``config.yaml`` gives these positions a smaller share of the risk budget.
"""

from __future__ import annotations

import pandas as pd

from indicators import ema, rolling_mean_volume
from portfolio import Position
from strategies.base import Action, Signal, Strategy


class CloseAuction(Strategy):
    name = "close_auction"

    def __init__(
        self,
        symbol: str = "",
        timeframe: str = "1Day",
        trend_ema: int = 20,
        volume_period: int = 20,
        volume_mult: float = 1.2,
        #: How near the high of the day the price must be, as a fraction of
        #: the day's range. 0.7 means the top 30%.
        close_strength: float = 0.7,
        atr_trail_mult: float = 2.0,
        atr_period: int = 14,
        hard_stop_atr_mult: float = 1.0,
        allow_short: bool = False,
        **params,
    ) -> None:
        super().__init__(
            symbol,
            timeframe,
            atr_period=atr_period,
            hard_stop_atr_mult=hard_stop_atr_mult,
            **params,
        )
        self.trend_ema = trend_ema
        self.volume_period = volume_period
        self.volume_mult = volume_mult
        self.close_strength = close_strength
        self.atr_trail_mult = atr_trail_mult
        self.allow_short = allow_short

    @property
    def warmup(self) -> int:
        return max(self.trend_ema, self.volume_period, self.atr_period) + 2

    def evaluate(self, window: pd.DataFrame, position: Position | None = None) -> Signal:
        if len(window) < self.warmup:
            return self._hold(window, "warming up")

        price = float(window["close"].iloc[-1])
        high = float(window["high"].iloc[-1])
        low = float(window["low"].iloc[-1])
        atr_value = self._atr(window)

        # The exit is a time, not a signal: the engine closes this the next
        # morning. Holding here keeps the trailing stop ratcheting in the
        # meantime, so an overnight collapse still meets a stop when the bot
        # is running.
        if position is not None:
            return self._hold(window, "holding until the planned morning exit")

        if atr_value <= 0:
            return self._hold(window, "ATR unavailable")

        day_range = high - low
        if day_range <= 0:
            return self._hold(window, "no range today")
        strength = (price - low) / day_range
        if strength < self.close_strength:
            return self._hold(
                window,
                f"closing at {strength:.0%} of today's range, "
                f"needs {self.close_strength:.0%}",
            )

        trend = ema(window["close"], self.trend_ema).iloc[-1]
        if pd.isna(trend):
            return self._hold(window, "trend EMA not established")
        if price <= float(trend):
            return self._hold(
                window, f"below the {self.trend_ema}-day EMA {float(trend):,.0f}"
            )

        avg_volume = rolling_mean_volume(window, self.volume_period).shift(1).iloc[-1]
        volume = float(window["volume"].iloc[-1])
        if pd.isna(avg_volume) or float(avg_volume) <= 0:
            return self._hold(window, "volume average not established")
        vol_ratio = volume / float(avg_volume)
        if vol_ratio < self.volume_mult:
            return self._hold(
                window, f"volume {vol_ratio:.2f}x < {self.volume_mult}x"
            )

        return self._signal(
            window,
            Action.ENTER_LONG,
            f"strong close: {strength:.0%} of range, above the "
            f"{self.trend_ema}-day EMA, {vol_ratio:.2f}x volume",
            atr_value,
            trail_stop=price - self.atr_trail_mult * atr_value,
            meta={
                "close_strength": strength,
                "trend_ema": float(trend),
                "volume_ratio": vol_ratio,
            },
        )

    def update_trailing_stop(self, window: pd.DataFrame, position: Position) -> float | None:
        return self._ratchet(position, window, self.atr_trail_mult)
