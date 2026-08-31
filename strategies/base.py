"""Strategy interface.

A strategy is handed a *window* of bars that ends at the bar being evaluated
and returns a :class:`Signal`.  It never sees a future bar, it never places an
order and it never knows how big the account is - sizing and execution are the
job of ``risk/manager.py`` and ``broker.py``.

That split is what makes the no-lookahead guarantee structural rather than a
convention: ``Backtester`` passes ``df.iloc[:i+1]``, so bar ``i+1`` is simply
not reachable from inside ``evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from indicators import atr as atr_indicator
from portfolio import Position


class Action(str, Enum):
    HOLD = "HOLD"
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT = "EXIT"

    @property
    def is_entry(self) -> bool:
        return self in (Action.ENTER_LONG, Action.ENTER_SHORT)

    @property
    def side(self) -> str:
        if self is Action.ENTER_LONG:
            return "LONG"
        if self is Action.ENTER_SHORT:
            return "SHORT"
        return ""


@dataclass(frozen=True)
class Signal:
    """What a strategy saw on one bar."""

    symbol: str
    action: Action
    reason: str = ""
    #: Close of the signal bar.  Reference only - it is NOT a fill price.
    price_ref: float = 0.0
    atr: float = 0.0
    #: Distance from entry to the hard stop, in price units.
    stop_distance: float = 0.0
    #: Index of the signal bar within the window the strategy was given.
    bar_index: int = -1
    bar_time: pd.Timestamp | None = None
    trail_stop: float | None = None
    take_profit: float | None = None
    meta: dict[str, Any] | None = None

    @property
    def actionable(self) -> bool:
        return self.action is not Action.HOLD


class Strategy:
    """Base class holding the shared plumbing."""

    name: str = "base"

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        atr_period: int = 14,
        hard_stop_atr_mult: float = 1.0,
        **params: Any,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.atr_period = atr_period
        self.hard_stop_atr_mult = hard_stop_atr_mult
        self.params = params

    # -- helpers -----------------------------------------------------------

    @property
    def warmup(self) -> int:
        """Bars required before the strategy can emit anything but HOLD."""
        return self.atr_period + 1

    @property
    def window_bars(self) -> int | None:
        """Trailing bars this strategy actually reads, or None for all of them.

        A backtest hands ``evaluate`` a window that grows by one bar each step.
        For a strategy whose answer depends only on a fixed lookback, rescanning
        the whole history every bar makes the run quadratic -- on 3-minute bars
        over a few months that is the difference between seconds and hours.
        Declaring the window turns it back into linear work.

        None is the safe default and keeps the existing behaviour exactly: an
        EMA is defined over everything before it, so truncating its input would
        change the numbers, not just the runtime.
        """
        return None

    def _atr(self, window: pd.DataFrame) -> float:
        series = atr_indicator(window, self.atr_period)
        value = series.iloc[-1] if len(series) else float("nan")
        return float(value) if pd.notna(value) else 0.0

    def _hold(
        self, window: pd.DataFrame, reason: str, meta: dict[str, Any] | None = None
    ) -> Signal:
        return Signal(
            symbol=self.symbol,
            action=Action.HOLD,
            reason=reason,
            price_ref=float(window["close"].iloc[-1]) if len(window) else 0.0,
            bar_index=len(window) - 1,
            bar_time=window.index[-1] if len(window) else None,
            meta=meta,
        )

    def _signal(
        self,
        window: pd.DataFrame,
        action: Action,
        reason: str,
        atr_value: float,
        trail_stop: float | None = None,
        take_profit: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Signal:
        return Signal(
            symbol=self.symbol,
            action=action,
            reason=reason,
            price_ref=float(window["close"].iloc[-1]),
            atr=atr_value,
            stop_distance=atr_value * self.hard_stop_atr_mult,
            bar_index=len(window) - 1,
            bar_time=window.index[-1],
            trail_stop=trail_stop,
            take_profit=take_profit,
            meta=meta,
        )

    # -- interface ---------------------------------------------------------

    def evaluate(self, window: pd.DataFrame, position: Position | None = None) -> Signal:
        """Return the signal for the last bar of *window*."""
        raise NotImplementedError

    def update_trailing_stop(
        self, window: pd.DataFrame, position: Position
    ) -> float | None:
        """Ratchet the trailing stop.  Strategies without one return ``None``."""
        return None

    def _ratchet(
        self, position: Position, window: pd.DataFrame, atr_mult: float
    ) -> float | None:
        """Shared ATR trailing-stop ratchet, used by breakout and trend.

        The stop only ever moves in the favourable direction, which is what
        makes it a trailing stop rather than a moving target.
        """
        atr_value = self._atr(window)
        if atr_value <= 0:
            return position.trail_stop
        close = float(window["close"].iloc[-1])
        if position.is_long:
            candidate = close - atr_mult * atr_value
            if position.trail_stop is None:
                return candidate
            return max(position.trail_stop, candidate)
        candidate = close + atr_mult * atr_value
        if position.trail_stop is None:
            return candidate
        return min(position.trail_stop, candidate)
