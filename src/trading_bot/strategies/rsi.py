import pandas as pd
from .base import BaseStrategy, Signal, TradeSignal


class RSIStrategy(BaseStrategy):
    """RSI 과매수/과매도 전략"""

    def __init__(self, params: dict):
        super().__init__(params)
        self.period = int(params.get("rsi_period", 14))
        self.oversold = float(params.get("rsi_oversold", 30))
        self.overbought = float(params.get("rsi_overbought", 70))

    @property
    def name(self) -> str:
        return f"RSI ({self.period})"

    def analyze(self, df: pd.DataFrame) -> TradeSignal:
        close = df["close"]
        rsi = self._calc_rsi(close, self.period)
        current_rsi = float(rsi.iloc[-1])
        price = float(close.iloc[-1])

        if current_rsi < self.oversold:
            confidence = (self.oversold - current_rsi) / self.oversold
            return TradeSignal(Signal.BUY, min(confidence, 1.0), f"RSI 과매도 ({current_rsi:.1f})", price)

        if current_rsi > self.overbought:
            confidence = (current_rsi - self.overbought) / (100 - self.overbought)
            return TradeSignal(Signal.SELL, min(confidence, 1.0), f"RSI 과매수 ({current_rsi:.1f})", price)

        return TradeSignal(Signal.HOLD, 0.5, f"RSI 중립 ({current_rsi:.1f})", price)

    @staticmethod
    def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))
