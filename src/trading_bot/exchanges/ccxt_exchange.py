import ccxt
import pandas as pd
from loguru import logger
from .base import BaseExchange


class CCXTExchange(BaseExchange):
    """CCXT 라이브러리 기반 거래소 어댑터 (Binance, Upbit 등 지원)"""

    def __init__(self, exchange_name: str, api_keys: dict, testnet: bool = False, dry_run: bool = True):
        self.dry_run = dry_run
        exchange_class = getattr(ccxt, exchange_name)
        options = {}

        if exchange_name == "binance" and testnet:
            options["defaultType"] = "future"

        self._exchange = exchange_class({
            **api_keys,
            "options": options,
            "enableRateLimit": True,
        })

        if exchange_name == "binance" and testnet:
            self._exchange.set_sandbox_mode(True)

        logger.info(f"거래소 연결: {exchange_name} (dry_run={dry_run}, testnet={testnet})")

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        raw = self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def get_balance(self) -> dict:
        if self.dry_run:
            return {"total": {"USDT": 10000.0, "KRW": 10_000_000}}
        return self._exchange.fetch_balance()

    def get_ticker(self, symbol: str) -> dict:
        return self._exchange.fetch_ticker(symbol)

    def create_market_buy(self, symbol: str, amount: float) -> dict:
        if self.dry_run:
            ticker = self.get_ticker(symbol)
            price = ticker["last"]
            logger.info(f"[DRY RUN] 시장가 매수: {symbol} {amount} @ {price:,.2f}")
            return {"id": "dry-run", "symbol": symbol, "side": "buy", "amount": amount, "price": price}
        return self._exchange.create_market_buy_order(symbol, amount)

    def create_market_sell(self, symbol: str, amount: float) -> dict:
        if self.dry_run:
            ticker = self.get_ticker(symbol)
            price = ticker["last"]
            logger.info(f"[DRY RUN] 시장가 매도: {symbol} {amount} @ {price:,.2f}")
            return {"id": "dry-run", "symbol": symbol, "side": "sell", "amount": amount, "price": price}
        return self._exchange.create_market_sell_order(symbol, amount)

    def create_limit_buy(self, symbol: str, amount: float, price: float) -> dict:
        if self.dry_run:
            logger.info(f"[DRY RUN] 지정가 매수: {symbol} {amount} @ {price:,.2f}")
            return {"id": "dry-run", "symbol": symbol, "side": "buy", "amount": amount, "price": price}
        return self._exchange.create_limit_buy_order(symbol, amount, price)

    def create_limit_sell(self, symbol: str, amount: float, price: float) -> dict:
        if self.dry_run:
            logger.info(f"[DRY RUN] 지정가 매도: {symbol} {amount} @ {price:,.2f}")
            return {"id": "dry-run", "symbol": symbol, "side": "sell", "amount": amount, "price": price}
        return self._exchange.create_limit_sell_order(symbol, amount, price)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        if self.dry_run:
            logger.info(f"[DRY RUN] 주문 취소: {order_id}")
            return {"id": order_id}
        return self._exchange.cancel_order(order_id, symbol)

    def get_open_orders(self, symbol: str) -> list:
        if self.dry_run:
            return []
        return self._exchange.fetch_open_orders(symbol)
