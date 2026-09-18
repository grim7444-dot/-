import time
import sys
from loguru import logger

from .config import Config
from .exchanges import create_exchange, BaseExchange
from .strategies import create_strategy, Signal
from .risk_manager import RiskManager


class TradingBot:
    """메인 트레이딩 봇"""

    def __init__(self, config: Config):
        self.config = config
        self._setup_logging()
        self.exchange: BaseExchange = create_exchange(config)
        self.strategy = create_strategy(config)
        self.risk = RiskManager(
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            max_drawdown_pct=config.max_drawdown_pct,
            max_positions=config.max_positions,
        )
        self._running = False

    def _setup_logging(self):
        logger.remove()
        logger.add(sys.stderr, level=self.config.log_level, colorize=True,
                   format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
        logger.add(self.config.log_file, level="DEBUG", rotation="10 MB", retention="30 days",
                   encoding="utf-8")

    def start(self):
        mode = "DRY RUN" if self.config.dry_run else "실거래"
        logger.info(f"트레이딩 봇 시작 [{mode}]")
        logger.info(f"거래소: {self.config.exchange_name} | 심볼: {self.config.symbol} | 전략: {self.strategy.name}")
        self._running = True

        balance = self._get_quote_balance()
        self.risk.set_initial_balance(balance)
        logger.info(f"초기 잔고: {balance:,.2f}")

        try:
            while self._running:
                self._tick()
                self._sleep_to_next_candle()
        except KeyboardInterrupt:
            logger.info("봇 종료 (사용자 인터럽트)")

    def stop(self):
        self._running = False

    def _tick(self):
        symbol = self.config.symbol
        try:
            df = self.exchange.get_ohlcv(symbol, self.config.timeframe)
            current_price = float(df["close"].iloc[-1])

            # 기존 포지션 손절/익절 체크
            if symbol in self.risk.positions:
                should_close, reason = self.risk.should_close(symbol, current_price)
                if should_close:
                    self._close_position(symbol, current_price, reason)
                    return

            # 잔고 및 최대 낙폭 체크
            balance = self._get_quote_balance()
            if self.risk.update_drawdown(balance):
                logger.error("최대 낙폭 초과로 봇 중지")
                self.stop()
                return

            # 전략 시그널 분석
            signal = self.strategy.analyze(df)
            logger.info(f"[{symbol}] 현재가: {current_price:,.2f} | 시그널: {signal.signal.value} | {signal.reason}")

            if signal.signal == Signal.BUY and self.risk.can_open_position(symbol):
                self._open_long(symbol, current_price, balance)
            elif signal.signal == Signal.SELL and symbol in self.risk.positions:
                self._close_position(symbol, current_price, signal.reason)

        except Exception as e:
            logger.error(f"틱 처리 오류: {e}")

    def _open_long(self, symbol: str, price: float, balance: float):
        amount_quote = min(self.config.trade_amount, balance * 0.95)
        amount_base = amount_quote / price
        order = self.exchange.create_market_buy(symbol, amount_base)
        self.risk.open_position(symbol, "long", price, amount_base, order.get("id", ""))

    def _close_position(self, symbol: str, price: float, reason: str):
        pos = self.risk.positions.get(symbol)
        if not pos:
            return
        self.exchange.create_market_sell(symbol, pos.amount)
        pnl = (price - pos.entry_price) / pos.entry_price * 100
        logger.info(f"포지션 종료: {symbol} | 사유: {reason} | PnL: {pnl:+.2f}%")
        self.risk.close_position(symbol)

    def _get_quote_balance(self) -> float:
        balance = self.exchange.get_balance()
        quote = self.config.symbol.split("/")[1]
        total = balance.get("total", {})
        return float(total.get(quote, 0))

    def _sleep_to_next_candle(self):
        tf_seconds = {
            "1m": 60, "5m": 300, "15m": 900,
            "1h": 3600, "4h": 14400, "1d": 86400,
        }
        seconds = tf_seconds.get(self.config.timeframe, 3600)
        logger.debug(f"다음 캔들까지 {seconds}초 대기...")
        time.sleep(seconds)
