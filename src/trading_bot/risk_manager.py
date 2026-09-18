from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Position:
    symbol: str
    side: str            # "long" or "short"
    entry_price: float
    amount: float
    stop_loss: float
    take_profit: float
    order_id: str = ""


class RiskManager:
    """손절/익절 및 포트폴리오 리스크 관리"""

    def __init__(self, stop_loss_pct: float, take_profit_pct: float, max_drawdown_pct: float, max_positions: int):
        self.stop_loss_pct = stop_loss_pct / 100
        self.take_profit_pct = take_profit_pct / 100
        self.max_drawdown_pct = max_drawdown_pct / 100
        self.max_positions = max_positions
        self.positions: dict[str, Position] = {}
        self.initial_balance: float = 0.0
        self.peak_balance: float = 0.0

    def set_initial_balance(self, balance: float):
        self.initial_balance = balance
        self.peak_balance = balance

    def can_open_position(self, symbol: str) -> bool:
        if symbol in self.positions:
            logger.warning(f"{symbol} 포지션 이미 존재")
            return False
        if len(self.positions) >= self.max_positions:
            logger.warning(f"최대 포지션 수 초과 ({self.max_positions})")
            return False
        return True

    def open_position(self, symbol: str, side: str, entry_price: float, amount: float, order_id: str = "") -> Position:
        if side == "long":
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)

        pos = Position(symbol, side, entry_price, amount, stop_loss, take_profit, order_id)
        self.positions[symbol] = pos
        logger.info(
            f"포지션 오픈: {symbol} {side} @ {entry_price:,.2f} "
            f"| SL: {stop_loss:,.2f} | TP: {take_profit:,.2f}"
        )
        return pos

    def close_position(self, symbol: str):
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"포지션 종료: {symbol}")

    def should_close(self, symbol: str, current_price: float) -> tuple[bool, str]:
        """현재가 기준으로 손절/익절 여부 판단"""
        pos = self.positions.get(symbol)
        if not pos:
            return False, ""

        if pos.side == "long":
            if current_price <= pos.stop_loss:
                return True, f"손절 (SL: {pos.stop_loss:,.2f})"
            if current_price >= pos.take_profit:
                return True, f"익절 (TP: {pos.take_profit:,.2f})"
        else:
            if current_price >= pos.stop_loss:
                return True, f"손절 (SL: {pos.stop_loss:,.2f})"
            if current_price <= pos.take_profit:
                return True, f"익절 (TP: {pos.take_profit:,.2f})"

        return False, ""

    def update_drawdown(self, current_balance: float) -> bool:
        """최대 낙폭 초과 여부 확인 (초과 시 True 반환)"""
        self.peak_balance = max(self.peak_balance, current_balance)
        if self.peak_balance == 0:
            return False
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        if drawdown >= self.max_drawdown_pct:
            logger.error(f"최대 낙폭 초과: {drawdown*100:.2f}% >= {self.max_drawdown_pct*100:.2f}%")
            return True
        return False
