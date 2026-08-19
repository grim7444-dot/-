"""Intraday momentum on 3-minute bars (단타).

The entry is a momentum break, filtered twice:

* the close must exceed the *prior* N-bar high, on at least ``volume_mult``
  times the average volume of those bars -- the same channel logic the hourly
  breakout uses, at a faster cadence;
* the price must be above the session's own VWAP.

The VWAP filter is what separates this from ``Breakout`` run faster. Intraday,
VWAP is where the day's volume actually changed hands, and a break above a
short-term high while the stock sits below it is usually a bounce inside a
decline rather than the start of a move. VWAP resets each session, so it says
something about *today* rather than about a window that happens to be 60
minutes long.

Costs decide whether any of this is worth doing. A round trip on KRX pays
commission both ways, transaction tax on the sell and slippage on both fills --
call it 0.38% at the assumptions in ``config.yaml``. The strategy therefore
refuses to enter a stock whose bars are too small for a winning trade to clear
that. The comparison is against the move a *win* keeps, not against one bar:
the trail sits ``atr_trail_mult`` ATR below the high, so that is the scale of
what a good trade gives up at the end and roughly the scale of what it keeps.
Requiring ``min_edge_mult`` round trips out of that is the test. Comparing
costs to a single ATR instead would have barred four of the five stocks this
runs on, all of which move 9-16% a day.

Exits are the shared ATR trail plus the flat-out time in ``SessionRules`` --
the engine closes the position before the closing auction whatever this says.
"""

from __future__ import annotations

import pandas as pd

from indicators import rolling_max, rolling_mean_volume
from portfolio import Position
from strategies.base import Action, Signal, Strategy


def session_vwap(window: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, restarted each session.

    Cumulating across days would anchor today's filter to last week's prices,
    which is the opposite of what an intraday reference is for.
    """
    typical = (window["high"] + window["low"] + window["close"]) / 3.0
    turnover = typical * window["volume"]
    days = pd.Index(window.index).normalize()
    cum_turnover = turnover.groupby(days).cumsum()
    cum_volume = window["volume"].groupby(days).cumsum()
    return cum_turnover / cum_volume.replace(0.0, pd.NA)


class Scalping(Strategy):
    name = "scalping"

    def __init__(
        self,
        symbol: str = "",
        timeframe: str = "3Min",
        period: int = 20,
        volume_mult: float = 1.5,
        atr_trail_mult: float = 1.5,
        #: How many round trips a typical winning trade must be worth.
        min_edge_mult: float = 2.0,
        round_trip_cost_pct: float = 0.0038,
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
        self.period = period
        self.volume_mult = volume_mult
        self.atr_trail_mult = atr_trail_mult
        self.min_edge_mult = min_edge_mult
        self.round_trip_cost_pct = round_trip_cost_pct
        self.allow_short = allow_short

    #: 09:00-15:30 in three-minute steps.
    BARS_PER_SESSION = 130

    @property
    def warmup(self) -> int:
        return max(self.period, self.atr_period) + 2

    @property
    def window_bars(self) -> int:
        """Channel, ATR and today's VWAP -- nothing here looks back further.

        Two sessions of margin so the current one is always complete even when
        the window starts mid-day.
        """
        return max(self.period, self.atr_period) + 2 * self.BARS_PER_SESSION

    @property
    def min_atr_pct(self) -> float:
        """Smallest ATR, as a fraction of price, worth trading at these costs."""
        capture = max(self.atr_trail_mult, 0.1)
        return self.round_trip_cost_pct * self.min_edge_mult / capture

    def evaluate(self, window: pd.DataFrame, position: Position | None = None) -> Signal:
        if len(window) < self.warmup:
            return self._hold(window, "warming up")

        price = float(window["close"].iloc[-1])
        atr_value = self._atr(window)
        vwap = session_vwap(window)
        vwap_now = vwap.iloc[-1]

        prior_high = rolling_max(window["high"], self.period).shift(1).iloc[-1]
        avg_volume = rolling_mean_volume(window, self.period).shift(1).iloc[-1]
        if pd.isna(prior_high) or pd.isna(avg_volume) or pd.isna(vwap_now):
            return self._hold(window, "channel not established")

        prior_high = float(prior_high)
        avg_volume = float(avg_volume)
        vwap_now = float(vwap_now)
        volume = float(window["volume"].iloc[-1])
        vol_ratio = (volume / avg_volume) if avg_volume > 0 else 0.0

        # --- manage an open position ---------------------------------------
        # The ATR trail and the hard stop are enforced by the engine; the only
        # exit this adds is losing the session's own reference price, which is
        # the signal that the move being traded has stopped working.
        if position is not None:
            if position.is_long and price < vwap_now:
                return self._signal(
                    window,
                    Action.EXIT,
                    f"lost session VWAP {vwap_now:,.0f}",
                    atr_value,
                )
            return self._hold(window, f"holding, VWAP {vwap_now:,.0f}")

        # --- entries --------------------------------------------------------
        if atr_value <= 0:
            return self._hold(window, "ATR unavailable")

        atr_pct = atr_value / price if price > 0 else 0.0
        if atr_pct < self.min_atr_pct:
            return self._hold(
                window,
                f"too quiet to pay for itself: ATR {atr_pct:.2%} < "
                f"{self.min_atr_pct:.2%} (round trip costs {self.round_trip_cost_pct:.2%})",
            )

        if price <= vwap_now:
            return self._hold(window, f"below session VWAP {vwap_now:,.0f}")
        if price <= prior_high:
            return self._hold(window, f"inside {self.period}-bar channel")
        if not (avg_volume > 0 and vol_ratio >= self.volume_mult):
            return self._hold(
                window,
                f"break rejected: volume {vol_ratio:.2f}x < {self.volume_mult}x",
            )

        return self._signal(
            window,
            Action.ENTER_LONG,
            f"3-min break of {prior_high:,.0f} on {vol_ratio:.2f}x volume, "
            f"above VWAP {vwap_now:,.0f}",
            atr_value,
            trail_stop=price - self.atr_trail_mult * atr_value,
            meta={
                "prior_high": prior_high,
                "vwap": vwap_now,
                "volume_ratio": vol_ratio,
                "atr_pct": atr_pct,
            },
        )

    def update_trailing_stop(self, window: pd.DataFrame, position: Position) -> float | None:
        return self._ratchet(position, window, self.atr_trail_mult)
