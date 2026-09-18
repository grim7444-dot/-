from abc import ABC, abstractmethod
import pandas as pd


class BaseExchange(ABC):
    """거래소 공통 인터페이스"""

    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """OHLCV 캔들 데이터 조회"""

    @abstractmethod
    def get_balance(self) -> dict:
        """잔고 조회"""

    @abstractmethod
    def get_ticker(self, symbol: str) -> dict:
        """현재가 조회"""

    @abstractmethod
    def create_market_buy(self, symbol: str, amount: float) -> dict:
        """시장가 매수"""

    @abstractmethod
    def create_market_sell(self, symbol: str, amount: float) -> dict:
        """시장가 매도"""

    @abstractmethod
    def create_limit_buy(self, symbol: str, amount: float, price: float) -> dict:
        """지정가 매수"""

    @abstractmethod
    def create_limit_sell(self, symbol: str, amount: float, price: float) -> dict:
        """지정가 매도"""

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """주문 취소"""

    @abstractmethod
    def get_open_orders(self, symbol: str) -> list:
        """미체결 주문 조회"""
