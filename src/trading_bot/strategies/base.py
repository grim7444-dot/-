from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import pandas as pd


class Signal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradeSignal:
    signal: Signal
    confidence: float      # 0.0 ~ 1.0
    reason: str
    price: float


class BaseStrategy(ABC):
    """트레이딩 전략 기본 클래스"""

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> TradeSignal:
        """OHLCV 데이터 분석 후 매매 시그널 반환"""

    @property
    @abstractmethod
    def name(self) -> str:
        """전략 이름"""
