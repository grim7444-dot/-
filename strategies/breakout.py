"""Volume-confirmed breakout for BTC/USD (1-hour bars).

Track the 20-period high and low.  A close above the *prior* 20-period high
goes long, a close below the *prior* 20-period low goes short (or closes a
long) - but only when volume is at least 1.5x its own 20-period average.  The
volume filter is what separates a real breakout from a drifting wick.

Positions carry a 2x ATR trailing stop on top of the mandatory 1% hard stop.

Note the ``.shift(1)`` on the channel: the current bar's own high is part of
``rolling_max`` by construction, so without the shift every bar that makes a
new high would "break out" of itself.
"""

from __future__ import annotations

import pandas as pd

from indicators import rolling_max, rolling_mean_volume, rolling_min
from portfolio import Position
from strategies.base import Action, Signal, Strategy


class Breakout(Strategy):
    name = "breakout"

    def __init__(
        self,
        symbol: str = "BTC/USD",
        timeframe: str = "1Hour",
        period: int = 20,
        volume_mult: float = 1.5,
        atr_trail_mult: float = 2.0,
        atr_period: int = 14,
        hard_stop_atr_mult: float = 1.0,
        allow_short: bool = True,
        **params,
    ) -> None:
        super().__init__(
            symbol,
            timeframe,
            atr_period=atr_period,
            hard_stop_atr_mult=hard_stop_atr_mult,
            **params,
        )
        self.period = period
        self.volume_mult = volume_mult
        self.atr_trail_mult = atr_trail_mult
        self.allow_short = allow_short

    @property
    def warmup(self) -> int:
        return max(self.period, self.atr_period) + 2

    def evaluate(self, window: pd.DataFrame, position: Position | None = None) -> Signal:
        if len(window) < self.warmup:
            return self._hold(window, "warming up")

        close = window["close"]
        price = float(close.iloc[-1])

        # Prior channel: exclude the bar being evaluated.
        prior_high = rolling_max(window["high"], self.period).shift(1).iloc[-1]
        prior_low = rolling_min(window["low"], self.period).shift(1).iloc[-1]
        avg_volume = rolling_mean_volume(window, self.period).shift(1).iloc[-1]
        volume = float(window["volume"].iloc[-1])

        if pd.isna(prior_high) or pd.isna(prior_low) or pd.isna(avg_volume):
            return self._hold(window, "channel not established")

        prior_high = float(prior_high)
        prior_low = float(prior_low)
        avg_volume = float(avg_volume)
        volume_ok = avg_volume > 0 and volume >= self.volume_mult * avg_volume
        vol_ratio = (volume / avg_volume) if avg_volume > 0 else 0.0

        atr_value = self._atr(window)
        breaks_up = price > prior_high
        breaks_down = price < prior_low

        # --- manage an open position ---------------------------------------
        if position is not None:
            if position.is_long and breaks_down and volume_ok:
                return self._signal(
                    window,
                    Action.EXIT,
                    f"broke {self.period}-bar low {prior_low:.2f} on {vol_ratio:.2f}x volume",
                    atr_value,
                )
            if position.is_short and breaks_up and volume_ok:
                return self._signal(
                    window,
                    Action.EXIT,
                    f"broke {self.period}-bar high {prior_high:.2f} on {vol_ratio:.2f}x volume",
                    atr_value,
                )
            return self._hold(window, f"holding, volume {vol_ratio:.2f}x")

        # --- entries --------------------------------------------------------
        if atr_value <= 0:
            return self._hold(window, "ATR unavailable")

        meta = {
            "prior_high": prior_high,
            "prior_low": prior_low,
            "volume_ratio": vol_ratio,
            "volume_mult": self.volume_mult,
        }

        if breaks_up and volume_ok:
            return self._signal(
                window,
                Action.ENTER_LONG,
                f"close {price:.2f} > {self.period}-bar high {prior_high:.2f} "
                f"on {vol_ratio:.2f}x volume",
                atr_value,
                trail_stop=price - self.atr_trail_mult * atr_value,
                meta=meta,
            )
        if breaks_down and volume_ok and self.allow_short:
            return self._signal(
                window,
                Action.ENTER_SHORT,
                f"close {price:.2f} < {self.period}-bar low {prior_low:.2f} "
                f"on {vol_ratio:.2f}x volume",
                atr_value,
                trail_stop=price + self.atr_trail_mult * atr_value,
                meta=meta,
            )

        if (breaks_up or breaks_down) and not volume_ok:
            return self._hold(
                window, f"breakout rejected: volume {vol_ratio:.2f}x < {self.volume_mult}x"
            )
        return self._hold(window, "inside channel")

    def update_trailing_stop(self, window: pd.DataFrame, position: Position) -> float | None:
        return self._ratchet(position, window, self.atr_trail_mult)
