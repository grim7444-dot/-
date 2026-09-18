import math
import sys
import time
import numpy as np
from datetime import datetime
from loguru import logger
from .config import Config
from .exchanges.mock_exchange import MockExchange
from .strategies import create_strategy, Signal
from .risk_manager import RiskManager


class BacktestRunner:
    """모의 거래소로 전략을 빠르게 시뮬레이션"""

    def __init__(self, config: Config, ticks: int = 50, speed: float = 0.3):
        self.config = config
        self.ticks = ticks
        self.speed = speed  # 틱 간 대기 시간(초)
        self.exchange = MockExchange(symbol=config.symbol)
        self.strategy = create_strategy(config)
        self.risk = RiskManager(
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            max_drawdown_pct=config.max_drawdown_pct,
            max_positions=config.max_positions,
        )
        self._trades: list[dict] = []
        self._equity: list[float] = []

    def run(self):
        logger.info("=" * 60)
        logger.info(f"  백테스트 시작: {self.strategy.name}")
        logger.info(f"  심볼: {self.config.symbol}  |  타임프레임: {self.config.timeframe}")
        logger.info(f"  전략: {self.strategy.name}  |  총 틱: {self.ticks}")
        logger.info("=" * 60)

        initial = self.exchange._balance_quote
        self.risk.set_initial_balance(initial)

        for tick in range(1, self.ticks + 1):
            self._tick(tick)
            time.sleep(self.speed)

        self._report(initial)

    def _tick(self, tick: int):
        symbol = self.config.symbol
        df = self.exchange.get_ohlcv(symbol, self.config.timeframe)
        current_price = float(df["close"].iloc[-1])

        # 손절/익절 체크
        if symbol in self.risk.positions:
            should_close, reason = self.risk.should_close(symbol, current_price)
            if should_close:
                self._close(symbol, current_price, reason)
                self._equity.append(self.exchange._balance_quote)
                return

        signal = self.strategy.analyze(df)
        balance = self.exchange._balance_quote

        indicator = {"buy": "▲", "sell": "▼", "hold": "─"}[signal.signal.value]
        logger.info(
            f"[틱 {tick:03d}] {current_price:>10,.2f} USDT  {indicator} {signal.signal.value.upper():<4}  "
            f"| {signal.reason}  | 잔고: ${balance:,.2f}"
        )

        if signal.signal == Signal.BUY and self.risk.can_open_position(symbol):
            amount = min(self.config.trade_amount, balance * 0.95) / current_price
            order = self.exchange.create_market_buy(symbol, amount)
            self.risk.open_position(symbol, "long", current_price, amount, order["id"])
            self._trades.append({"type": "buy", "price": current_price, "tick": tick})

        elif signal.signal == Signal.SELL and symbol in self.risk.positions:
            self._close(symbol, current_price, signal.reason)
            self._trades.append({"type": "sell", "price": current_price, "tick": tick})

        self._equity.append(self.exchange._balance_quote)

    def _close(self, symbol: str, price: float, reason: str):
        pos = self.risk.positions.get(symbol)
        if not pos:
            return
        self.exchange.create_market_sell(symbol, pos.amount)
        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
        logger.info(f"  → 포지션 종료: {reason}  PnL {pnl_pct:+.2f}%")
        self.risk.close_position(symbol)

    def _report(self, initial: float):
        final = self.exchange._balance_quote
        total_return = (final - initial) / initial * 100
        buys = sum(1 for t in self._trades if t["type"] == "buy")
        sells = sum(1 for t in self._trades if t["type"] == "sell")

        logger.info("")
        logger.info("=" * 60)
        logger.info("  백테스트 결과")
        logger.info("=" * 60)
        logger.info(f"  초기 자본  : ${initial:>10,.2f}")
        logger.info(f"  최종 자본  : ${final:>10,.2f}")
        logger.info(f"  총 수익률  : {total_return:>+.2f}%")
        logger.info(f"  매수 횟수  : {buys}회")
        logger.info(f"  매도 횟수  : {sells}회")
        if self._equity:
            peak = max(self._equity)
            max_dd = min((e - peak) / peak * 100 for e in self._equity) if peak > 0 else 0
            logger.info(f"  최대 낙폭  : {max_dd:.2f}%")
        logger.info("=" * 60)
