import random
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from .base import BaseExchange


class MockExchange(BaseExchange):
    """백테스트용 모의 거래소 — 랜덤 워크로 가격 생성"""

    def __init__(self, symbol: str = "BTC/USDT", initial_price: float = 50000.0, seed: int = 42):
        self.symbol = symbol
        self._price = initial_price
        self._balance_quote = 10000.0  # 초기 자본
        self._holdings: dict[str, float] = {}
        rng = np.random.default_rng(seed)
        self._rng = rng

    def _next_price(self) -> float:
        drift = 0.0001
        volatility = 0.008
        shock = self._rng.normal(drift, volatility)
        self._price *= math.exp(shock)
        return self._price

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """과거 캔들 생성 (현재 가격에서 역산)"""
        now = datetime.utcnow()
        tf_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        minutes = tf_min.get(timeframe, 60)

        # 현재 self._price 기준으로 과거 limit개 캔들 생성
        prices = []
        p = self._price
        for _ in range(limit):
            shock = self._rng.normal(0, 0.005)
            p_open = p
            p_close = p * math.exp(shock)
            p_high = max(p_open, p_close) * (1 + abs(self._rng.normal(0, 0.002)))
            p_low = min(p_open, p_close) * (1 - abs(self._rng.normal(0, 0.002)))
            vol = float(self._rng.uniform(100, 1000))
            prices.append((p_open, p_high, p_low, p_close, vol))
            p = p_close

        prices.reverse()
        idx = [now - timedelta(minutes=minutes * (limit - i - 1)) for i in range(limit)]
        df = pd.DataFrame(prices, columns=["open", "high", "low", "close", "volume"], index=idx)
        df.index.name = "timestamp"
        return df

    def get_balance(self) -> dict:
        return {"total": {"USDT": self._balance_quote, "KRW": self._balance_quote * 1300}}

    def get_ticker(self, symbol: str) -> dict:
        price = self._next_price()
        return {"symbol": symbol, "last": price, "bid": price * 0.9995, "ask": price * 1.0005}

    def create_market_buy(self, symbol: str, amount: float) -> dict:
        price = self._next_price()
        cost = price * amount
        self._balance_quote -= cost
        base = symbol.split("/")[0]
        self._holdings[base] = self._holdings.get(base, 0) + amount
        logger.info(f"[MOCK] 매수 체결: {amount:.6f} {base} @ {price:,.2f} (잔고: ${self._balance_quote:,.2f})")
        return {"id": f"mock-{id(self)}", "symbol": symbol, "side": "buy", "amount": amount, "price": price}

    def create_market_sell(self, symbol: str, amount: float) -> dict:
        price = self._next_price()
        proceeds = price * amount
        self._balance_quote += proceeds
        base = symbol.split("/")[0]
        self._holdings[base] = max(0, self._holdings.get(base, 0) - amount)
        logger.info(f"[MOCK] 매도 체결: {amount:.6f} {base} @ {price:,.2f} (잔고: ${self._balance_quote:,.2f})")
        return {"id": f"mock-{id(self)}", "symbol": symbol, "side": "sell", "amount": amount, "price": price}

    def create_limit_buy(self, symbol: str, amount: float, price: float) -> dict:
        return self.create_market_buy(symbol, amount)

    def create_limit_sell(self, symbol: str, amount: float, price: float) -> dict:
        return self.create_market_sell(symbol, amount)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        return {"id": order_id}

    def get_open_orders(self, symbol: str) -> list:
        return []
