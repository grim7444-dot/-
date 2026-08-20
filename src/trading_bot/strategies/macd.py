import pandas as pd
from .base import BaseStrategy, Signal, TradeSignal


class MACDStrategy(BaseStrategy):
    """MACD 전략"""

    def __init__(self, params: dict):
        super().__init__(params)
        self.fast = int(params.get("macd_fast", 12))
        self.slow = int(params.get("macd_slow", 26))
        self.signal_period = int(params.get("macd_signal", 9))

    @property
    def name(self) -> str:
        return f"MACD ({self.fast}/{self.slow}/{self.signal_period})"

    def analyze(self, df: pd.DataFrame) -> TradeSignal:
        close = df["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        prev_hist = float(histogram.iloc[-2])
        curr_hist = float(histogram.iloc[-1])
        price = float(close.iloc[-1])

        # 히스토그램이 음수에서 양수로 전환 (매수)
        if prev_hist < 0 and curr_hist >= 0:
            return TradeSignal(Signal.BUY, 0.75, f"MACD 상향 돌파 (hist={curr_hist:.4f})", price)

        # 히스토그램이 양수에서 음수로 전환 (매도)
        if prev_hist > 0 and curr_hist <= 0:
            return TradeSignal(Signal.SELL, 0.75, f"MACD 하향 돌파 (hist={curr_hist:.4f})", price)

        return TradeSignal(Signal.HOLD, 0.5, f"MACD 유지 (hist={curr_hist:.4f})", price)
