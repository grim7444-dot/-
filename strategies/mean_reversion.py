"""Mean reversion for SPY and QQQ (15-minute bars).

Compute a 20-period simple moving average and standard deviation of the close.
Go long when price sits ``entry_sigma`` standard deviations *below* the mean,
short when it sits the same distance *above*, and flatten as soon as price
comes back to the mean.

SPY uses 1.5 sigma.  QQQ uses 1.8 sigma - it is the more volatile of the two,
so the band is widened to keep the entry count comparable rather than letting
QQQ trade on ordinary noise.
"""

from __future__ import annotations

import pandas as pd

from indicators import rolling_std, sma
from portfolio import Position
from strategies.base import Action, Signal, Strategy


class MeanReversion(Strategy):
    name = "mean_reversion"

    def __init__(
        self,
        symbol: str,
        timeframe: str = "15Min",
        period: int = 20,
        entry_sigma: float = 1.5,
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
        self.entry_sigma = entry_sigma
        self.allow_short = allow_short

    @property
    def warmup(self) -> int:
        return max(self.period, self.atr_period) + 1

    def evaluate(self, window: pd.DataFrame, position: Position | None = None) -> Signal:
        if len(window) < self.warmup:
            return self._hold(window, "warming up")

        close = window["close"]
        mean_series = sma(close, self.period)
        std_series = rolling_std(close, self.period)

        mean = mean_series.iloc[-1]
        std = std_series.iloc[-1]
        price = float(close.iloc[-1])

        if pd.isna(mean) or pd.isna(std) or std <= 0:
            return self._hold(window, "no dispersion in window")

        mean = float(mean)
        std = float(std)
        z = (price - mean) / std

        # --- exit first: reverting to the mean closes whatever we hold ----
        if position is not None:
            prev_price = float(close.iloc[-2])
            prev_mean = float(mean_series.iloc[-2]) if pd.notna(mean_series.iloc[-2]) else mean
            if position.is_long:
                # Crossed up through the mean (or is sitting on it).
                if price >= mean and prev_price < prev_mean:
                    return self._signal(
                        window,
                        Action.EXIT,
                        f"reverted to mean {mean:.4f} (z={z:+.2f})",
                        self._atr(window),
                    )
            elif position.is_short:
                if price <= mean and prev_price > prev_mean:
                    return self._signal(
                        window,
                        Action.EXIT,
                        f"reverted to mean {mean:.4f} (z={z:+.2f})",
                        self._atr(window),
                    )
            return self._hold(window, f"holding, z={z:+.2f}")

        # --- entries -------------------------------------------------------
        atr_value = self._atr(window)
        if atr_value <= 0:
            return self._hold(window, "ATR unavailable")

        meta = {"z": z, "mean": mean, "std": std, "sigma": self.entry_sigma}
        if z <= -self.entry_sigma:
            return self._signal(
                window,
                Action.ENTER_LONG,
                f"z={z:+.2f} <= -{self.entry_sigma} (stretched below {self.period}-SMA)",
                atr_value,
                meta=meta,
            )
        if z >= self.entry_sigma and self.allow_short:
            return self._signal(
                window,
                Action.ENTER_SHORT,
                f"z={z:+.2f} >= +{self.entry_sigma} (stretched above {self.period}-SMA)",
                atr_value,
                meta=meta,
            )
        return self._hold(window, f"inside band, z={z:+.2f}")
