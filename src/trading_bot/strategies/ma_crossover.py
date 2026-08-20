import pandas as pd
from .base import BaseStrategy, Signal, TradeSignal


class MACrossoverStrategy(BaseStrategy):
    """이동평균 크로스오버 전략 (골든/데드 크로스)"""

    def __init__(self, params: dict):
        super().__init__(params)
        self.fast = int(params.get("fast_period", 9))
        self.slow = int(params.get("slow_period", 21))

    @property
    def name(self) -> str:
        return f"MA Crossover ({self.fast}/{self.slow})"

    def analyze(self, df: pd.DataFrame) -> TradeSignal:
        close = df["close"]
        fast_ma = close.ewm(span=self.fast, adjust=False).mean()
        slow_ma = close.ewm(span=self.slow, adjust=False).mean()

        prev_fast, curr_fast = fast_ma.iloc[-2], fast_ma.iloc[-1]
        prev_slow, curr_slow = slow_ma.iloc[-2], slow_ma.iloc[-1]
        price = float(close.iloc[-1])

        # 골든 크로스: 단기 MA가 장기 MA를 상향 돌파
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return TradeSignal(Signal.BUY, 0.8, "골든 크로스 발생", price)

        # 데드 크로스: 단기 MA가 장기 MA를 하향 돌파
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return TradeSignal(Signal.SELL, 0.8, "데드 크로스 발생", price)

        return TradeSignal(Signal.HOLD, 0.5, "추세 유지", price)
