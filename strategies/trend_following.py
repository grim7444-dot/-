"""EMA-crossover trend following for GLD and USO (4-hour bars).

A 50-period EMA crossing above the 200-period EMA opens a long; crossing back
below closes it and (when shorting is enabled) opens a short.  Positions carry
a 3x ATR trailing stop on top of the mandatory 1% hard stop.

GLD and USO are **ETF proxies** for gold and crude oil, not futures contracts.
Their bars carry ETF mechanics - expense ratios, and for USO the roll cost of
the front-month futures it holds - so a signal here is a signal on the ETF,
not on the underlying commodity.  See README.
"""

from __future__ import annotations

import pandas as pd

from indicators import ema
from portfolio import Position
from strategies.base import Action, Signal, Strategy


class TrendFollowing(Strategy):
    name = "trend_following"

    def __init__(
        self,
        symbol: str,
        timeframe: str = "4Hour",
        fast_ema: int = 50,
        slow_ema: int = 200,
        atr_trail_mult: float = 3.0,
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
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_trail_mult = atr_trail_mult
        self.allow_short = allow_short

    @property
    def warmup(self) -> int:
        return max(self.slow_ema, self.atr_period) + 2

    def evaluate(self, window: pd.DataFrame, position: Position | None = None) -> Signal:
        if len(window) < self.warmup:
            return self._hold(window, "warming up")

        close = window["close"]
        fast = ema(close, self.fast_ema)
        slow = ema(close, self.slow_ema)

        if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]) or pd.isna(slow.iloc[-2]):
            return self._hold(window, "EMAs not established")

        fast_now, fast_prev = float(fast.iloc[-1]), float(fast.iloc[-2])
        slow_now, slow_prev = float(slow.iloc[-1]), float(slow.iloc[-2])

        golden_cross = fast_prev <= slow_prev and fast_now > slow_now
        death_cross = fast_prev >= slow_prev and fast_now < slow_now

        atr_value = self._atr(window)
        price = float(close.iloc[-1])

        # --- manage an open position ---------------------------------------
        if position is not None:
            if position.is_long and death_cross:
                return self._signal(
                    window,
                    Action.EXIT,
                    f"{self.fast_ema} EMA crossed below {self.slow_ema} EMA",
                    atr_value,
                )
            if position.is_short and golden_cross:
                return self._signal(
                    window,
                    Action.EXIT,
                    f"{self.fast_ema} EMA crossed above {self.slow_ema} EMA",
                    atr_value,
                )
            return self._hold(window, "holding trend")

        # --- entries --------------------------------------------------------
        if atr_value <= 0:
            return self._hold(window, "ATR unavailable")

        meta = {"fast": fast_now, "slow": slow_now}
        if golden_cross:
            return self._signal(
                window,
                Action.ENTER_LONG,
                f"golden cross: {self.fast_ema} EMA {fast_now:.4f} > "
                f"{self.slow_ema} EMA {slow_now:.4f}",
                atr_value,
                trail_stop=price - self.atr_trail_mult * atr_value,
                meta=meta,
            )
        if death_cross and self.allow_short:
            return self._signal(
                window,
                Action.ENTER_SHORT,
                f"death cross: {self.fast_ema} EMA {fast_now:.4f} < "
                f"{self.slow_ema} EMA {slow_now:.4f}",
                atr_value,
                trail_stop=price + self.atr_trail_mult * atr_value,
                meta=meta,
            )
        return self._hold(window, "no cross")

    def update_trailing_stop(self, window: pd.DataFrame, position: Position) -> float | None:
        return self._ratchet(position, window, self.atr_trail_mult)
