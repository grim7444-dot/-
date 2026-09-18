import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_bot.risk_manager import RiskManager


@pytest.fixture
def rm():
    return RiskManager(stop_loss_pct=2.0, take_profit_pct=4.0, max_drawdown_pct=10.0, max_positions=3)


def test_open_and_close_position(rm):
    rm.open_position("BTC/USDT", "long", 50000.0, 0.01)
    assert "BTC/USDT" in rm.positions
    rm.close_position("BTC/USDT")
    assert "BTC/USDT" not in rm.positions


def test_stop_loss_triggered(rm):
    rm.open_position("BTC/USDT", "long", 50000.0, 0.01)
    close, reason = rm.should_close("BTC/USDT", 48000.0)  # -4% > -2% SL
    assert close is True
    assert "손절" in reason


def test_take_profit_triggered(rm):
    rm.open_position("BTC/USDT", "long", 50000.0, 0.01)
    close, reason = rm.should_close("BTC/USDT", 53000.0)  # +6% > +4% TP
    assert close is True
    assert "익절" in reason


def test_hold_within_range(rm):
    rm.open_position("BTC/USDT", "long", 50000.0, 0.01)
    close, _ = rm.should_close("BTC/USDT", 50500.0)  # +1%, 범위 내
    assert close is False


def test_max_positions(rm):
    for i in range(3):
        rm.open_position(f"COIN{i}/USDT", "long", 1000.0, 1.0)
    assert not rm.can_open_position("COIN4/USDT")


def test_max_drawdown(rm):
    rm.set_initial_balance(10000.0)
    assert rm.update_drawdown(9100.0) is False   # -9%, 아직 OK
    assert rm.update_drawdown(8900.0) is True    # -11% from peak, 초과
