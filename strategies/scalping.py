"""Intraday momentum on intraday bars, flat by the close (단타).

The entry is a momentum break, filtered twice:

* the close must exceed the *prior* N-bar high, on at least ``volume_mult``
  times the average volume of those bars;
* the price must be above the session's own VWAP.

The VWAP filter is what separates this from ``Breakout`` run faster. Intraday,
VWAP is where the day's volume actually changed hands, and a break above a
short-term high while the stock sits below it is usually a bounce inside a
decline rather than the start of a move. VWAP resets each session, so it says
something about *today* rather than about a fixed-length window.

Costs decide the timeframe before anything else does. A round trip on KRX pays
commission both ways, transaction tax on the sell and slippage on both fills --
0.38% at the assumptions in ``config.yaml``. The useful way to read that is as
a share of what the trade risks: the hard stop is one ATR, so a trade risks
``ATR`` and pays ``0.38%`` to do it, and the ratio ``0.38% / ATR%`` is what
has to be small. Measured on these five stocks:

    3-minute bars   ATR 0.34%   costs are 112% of the amount risked
    5-minute        ATR 0.53%    72%
    10-minute       ATR 0.91%    42%
    15-minute       ATR 1.20%    32%
    30-minute       ATR 1.93%    20%
    60-minute       ATR 3.13%    12%

A breakout system does not earn 0.42 R per trade, so anything faster than
roughly half-hourly loses to its own friction whatever the entry rule is.
``max_cost_share`` is that ratio, and it is checked before every entry.

These numbers came from ``profile --atr-sweep``, which measures each
timeframe. An earlier version of this file estimated them from daily ATR by
the square root of time, overstated 3-minute volatility by a factor of three,
and set a threshold that silently refused every entry for a whole session.

Exits are the shared ATR trail plus the flat-out time in ``SessionRules`` --
the engine closes the position before the closing auction whatever this says.
"""

from __future__ import annotations

import pandas as pd

from indicators import bar_strength, rolling_max, rolling_mean_volume
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
        timeframe: str = "30Min",
        period: int = 10,
        volume_mult: float = 1.5,
        atr_trail_mult: float = 1.5,
        #: Largest share of the risked amount that costs may eat. The hard
        #: stop is one ATR, so this caps `round_trip_cost_pct / ATR%`.
        max_cost_share: float = 0.25,
        round_trip_cost_pct: float = 0.0038,
        atr_period: int = 14,
        hard_stop_atr_mult: float = 1.0,
        allow_short: bool = False,
        #: Fixed stop as a fraction of price (e.g. 0.02 = 2%). When set,
        #: overrides the ATR-based stop and cost gate.
        stop_pct: float = 0.0,
        #: Fixed take-profit as a fraction of price (e.g. 0.04 = 4%).
        take_profit_pct: float = 0.0,
        #: Minimum bar strength on the breakout bar: close must be this far
        #: into the bar's range (0=low, 1=high). 0.5 means the bar must close
        #: in the upper half -- i.e. buyers controlled the bar, not sellers.
        min_bar_strength: float = 0.0,
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
        self.max_cost_share = max_cost_share
        self.round_trip_cost_pct = round_trip_cost_pct
        self.allow_short = allow_short
        self.stop_pct = stop_pct
        self.take_profit_pct = take_profit_pct
        self.min_bar_strength = min_bar_strength

    #: Bars in one 09:00-15:30 session, by timeframe label.
    _SESSION_BARS = {
        "1Min": 390, "3Min": 130, "5Min": 78, "10Min": 39,
        "15Min": 26, "30Min": 13, "60Min": 7,
    }

    @property
    def warmup(self) -> int:
        return max(self.period, self.atr_period) + 2

    @property
    def window_bars(self) -> int:
        """Channel, ATR and today's VWAP -- nothing here looks back further.

        Two sessions of margin so the current one is always complete even when
        the window starts mid-day.
        """
        per_session = self._SESSION_BARS.get(self.timeframe, 130)
        return max(self.period, self.atr_period) + 2 * per_session

    @property
    def min_atr_pct(self) -> float:
        """Smallest ATR, as a fraction of price, worth trading at these costs."""
        return self.round_trip_cost_pct / max(self.max_cost_share, 0.01)

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
        if position is not None:
            if position.take_profit and position.is_long and price >= position.take_profit:
                return self._signal(
                    window, Action.EXIT,
                    f"take profit {position.take_profit:,.0f} reached (close={price:,.0f})",
                    atr_value,
                )
            if position.is_long and price < vwap_now:
                return self._signal(
                    window,
                    Action.EXIT,
                    f"lost session VWAP {vwap_now:,.0f}",
                    atr_value,
                )
            return self._hold(window, f"holding, VWAP {vwap_now:,.0f}")

        # --- entries --------------------------------------------------------
        # Fixed-pct stop overrides the ATR cost gate entirely.
        if self.stop_pct > 0:
            effective_atr = price * self.stop_pct
            cost_share = self.round_trip_cost_pct / self.stop_pct
            if cost_share > self.max_cost_share:
                return self._hold(
                    window,
                    f"too expensive: costs {cost_share:.0%} of stop (limit {self.max_cost_share:.0%})",
                )
        else:
            if atr_value <= 0:
                return self._hold(window, "ATR unavailable")
            atr_pct = atr_value / price if price > 0 else 0.0
            if atr_pct < self.min_atr_pct:
                return self._hold(
                    window,
                    f"too quiet to pay for itself: costs would be "
                    f"{self.round_trip_cost_pct / atr_pct:.0%} of the amount risked "
                    f"(ATR {atr_pct:.2%}, limit {self.max_cost_share:.0%})"
                    if atr_pct > 0
                    else "ATR unavailable",
                )
            effective_atr = atr_value

        if price <= vwap_now:
            return self._hold(window, f"below session VWAP {vwap_now:,.0f}")
        if price <= prior_high:
            return self._hold(window, f"inside {self.period}-bar channel")
        if not (avg_volume > 0 and vol_ratio >= self.volume_mult):
            return self._hold(
                window,
                f"break rejected: volume {vol_ratio:.2f}x < {self.volume_mult}x",
            )

        if self.min_bar_strength > 0:
            strength = float(bar_strength(window).iloc[-1])
            if strength < self.min_bar_strength:
                return self._hold(
                    window,
                    f"breakout bar closed weak: {strength:.0%} of range "
                    f"(need {self.min_bar_strength:.0%} -- sellers controlled bar)",
                )

        take_profit = price * (1 + self.take_profit_pct) if self.take_profit_pct > 0 else None
        stop_label = (
            f"{self.stop_pct:.0%} fixed stop" if self.stop_pct > 0
            else f"ATR trail"
        )
        tp_label = f", TP {take_profit:,.0f}" if take_profit else ""
        return self._signal(
            window,
            Action.ENTER_LONG,
            f"{self.timeframe} break of {prior_high:,.0f} on {vol_ratio:.2f}x volume, "
            f"above VWAP {vwap_now:,.0f} [{stop_label}{tp_label}]",
            effective_atr,
            trail_stop=price - self.atr_trail_mult * effective_atr if self.stop_pct == 0 else None,
            take_profit=take_profit,
            meta={
                "prior_high": prior_high,
                "vwap": vwap_now,
                "volume_ratio": vol_ratio,
            },
        )

    def update_trailing_stop(self, window: pd.DataFrame, position: Position) -> float | None:
        return self._ratchet(position, window, self.atr_trail_mult)
