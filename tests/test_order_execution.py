"""Marketable-limit order pricing (수수료/슬리피지 방지 주문 최적화).

User request (2026-08-27, Korean): 수수료 및 슬리피지 방지 주문 최적화. A
plain market order has no price protection -- in a thin book it can fill
meaningfully worse than the price the bot decided on. _marketable_limit_price
quotes past the best opposing quote by a small tick buffer: aggressive
enough to fill immediately (crosses the spread, same as a market order
would), but with an explicit ceiling/floor on how far the fill can drift.
"""

from __future__ import annotations

from dataclasses import dataclass

from market.rules import KOSDAQ, KOSPI, KrxRules
from portfolio import LONG, SHORT


@dataclass
class _FakeOrderBook:
    best_bid: float = 0.0
    best_ask: float = 0.0


RULES = KrxRules()


def test_buy_quotes_past_the_best_ask_by_the_tick_buffer():
    from main import _marketable_limit_price

    book = _FakeOrderBook(best_bid=9_990.0, best_ask=10_000.0)
    price = _marketable_limit_price(book, LONG, KOSPI, RULES, buffer_ticks=2)
    # tick at 10,000 KRW is 10 -- 2 ticks past the ask is 10,020.
    assert price == 10_020.0


def test_sell_quotes_past_the_best_bid_by_the_tick_buffer():
    from main import _marketable_limit_price

    book = _FakeOrderBook(best_bid=10_000.0, best_ask=10_010.0)
    price = _marketable_limit_price(book, SHORT, KOSPI, RULES, buffer_ticks=2)
    assert price == 9_980.0


def test_zero_buffer_quotes_exactly_at_the_best_opposing_price():
    from main import _marketable_limit_price

    book = _FakeOrderBook(best_bid=9_990.0, best_ask=10_000.0)
    assert _marketable_limit_price(book, LONG, KOSPI, RULES, buffer_ticks=0) == 10_000.0
    assert _marketable_limit_price(book, SHORT, KOSPI, RULES, buffer_ticks=0) == 9_990.0


def test_a_price_off_the_tick_grid_still_rounds_the_safe_direction():
    from main import _marketable_limit_price

    book = _FakeOrderBook(best_bid=9_990.0, best_ask=10_003.0)
    # base+buffer = 10,023 -- not on the 10-KRW grid at this band -- must
    # round UP (never down, which would fail to clear the ask).
    price = _marketable_limit_price(book, LONG, KOSPI, RULES, buffer_ticks=2)
    assert price == 10_030.0


def test_no_orderbook_falls_back_to_a_plain_market_order():
    """Missing quote data must never be a reason to refuse an otherwise-valid trade."""
    from main import _marketable_limit_price

    assert _marketable_limit_price(None, LONG, KOSPI, RULES, buffer_ticks=2) is None


def test_a_zero_quote_also_falls_back_to_market():
    from main import _marketable_limit_price

    book = _FakeOrderBook(best_bid=0.0, best_ask=0.0)
    assert _marketable_limit_price(book, LONG, KOSPI, RULES, buffer_ticks=2) is None
    assert _marketable_limit_price(book, SHORT, KOSPI, RULES, buffer_ticks=2) is None


def test_kosdaq_uses_its_own_tick_ladder():
    from main import _marketable_limit_price

    book = _FakeOrderBook(best_bid=9_990.0, best_ask=10_000.0)
    kospi_price = _marketable_limit_price(book, LONG, KOSPI, RULES, buffer_ticks=2)
    kosdaq_price = _marketable_limit_price(book, LONG, KOSDAQ, RULES, buffer_ticks=2)
    # Same inputs, routed through the market-specific tick ladder each time --
    # this only asserts they're each computed via their own market's rules,
    # not that they must differ (KOSPI/KOSDAQ ladders can coincide at a price).
    assert kospi_price == RULES.round_to_tick(10_020.0, KOSPI, "up")
    assert kosdaq_price == RULES.round_to_tick(10_020.0, KOSDAQ, "up")


# ---------------------------------------------------------------------------
# _submit_exit routing: a chosen take-profit uses a limit order, a stop or a
# forced time exit uses a plain market order regardless of the config toggle
# ---------------------------------------------------------------------------


class _FakeExecBroker:
    def __init__(self, orderbook=None):
        self._orderbook = orderbook
        self.orders: list[dict] = []

    def get_stock_info(self, code):
        from broker import StockInfo

        return StockInfo(code=code, tradable=True)

    def get_orderbook(self, code):
        return self._orderbook

    def submit_order(self, **kwargs):
        from broker import OrderResult

        self.orders.append(kwargs)
        return OrderResult(
            code=kwargs["code"], side=kwargs["side"], qty=kwargs["qty"], submitted=True, order_id="1",
        )


class _FakeExecNotifier:
    def send(self, *a, **k):
        pass

    def alert_exit(self, *a, **k):
        pass


def _exec_engine(portfolio, config, broker, use_limit_orders=True, limit_buffer_ticks=2):
    from main import TradingEngine
    from risk.manager import RiskManager

    class _FakeRTForExec:
        def name_of(self, code):
            return ""

    engine = TradingEngine.__new__(TradingEngine)
    rt = _FakeRTForExec()
    rt.config = config
    rt.portfolio = portfolio
    rt.risk = RiskManager(config, portfolio)
    rt.rules = RULES
    rt.broker = broker
    engine.rt = rt
    engine._last_exit_time = {}
    engine._last_exit_was_profit = {}
    engine._tg_notifier = _FakeExecNotifier()
    engine._use_limit_orders = use_limit_orders
    engine._limit_buffer_ticks = limit_buffer_ticks
    return engine


def _open_exec_position(portfolio, code="005930", entry_price=10_000.0):
    from portfolio import LONG, Position

    portfolio.open_position(
        Position(
            symbol=code, side=LONG, qty=10,
            entry_price=entry_price, stop_price=entry_price * 0.98, stop_distance=entry_price * 0.02,
        )
    )
    return portfolio.get(code)


def test_submit_exit_uses_a_limit_price_for_a_chosen_take_profit(portfolio, config):
    book = _FakeOrderBook(best_bid=10_000.0, best_ask=10_010.0)
    broker = _FakeExecBroker(orderbook=book)
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 10_050.0, "확정 익절", [], {}, True, "",
        market=KOSPI, use_limit=True,
    )

    assert len(broker.orders) == 1
    assert broker.orders[0]["price"] == 9_980.0  # sell: best_bid - 2 ticks


def test_submit_exit_uses_a_market_order_when_use_limit_is_false(portfolio, config):
    """This is the hard-stop / forced-time-exit path -- it must always ignore the orderbook."""
    book = _FakeOrderBook(best_bid=10_000.0, best_ask=10_010.0)
    broker = _FakeExecBroker(orderbook=book)
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 9_800.0, "stop hit", [], {}, True, "",
        market=KOSPI, use_limit=False,
    )

    assert len(broker.orders) == 1
    assert broker.orders[0]["price"] is None


def test_submit_exit_falls_back_to_market_with_no_orderbook_even_when_use_limit_is_true(portfolio, config):
    broker = _FakeExecBroker(orderbook=None)
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 10_050.0, "확정 익절", [], {}, True, "",
        market=KOSPI, use_limit=True,
    )

    assert broker.orders[0]["price"] is None


def test_submit_exit_respects_the_global_use_limit_orders_toggle(portfolio, config):
    book = _FakeOrderBook(best_bid=10_000.0, best_ask=10_010.0)
    broker = _FakeExecBroker(orderbook=book)
    engine = _exec_engine(portfolio, config, broker, use_limit_orders=False)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 10_050.0, "확정 익절", [], {}, True, "",
        market=KOSPI, use_limit=True,
    )

    assert broker.orders[0]["price"] is None


# ---------------------------------------------------------------------------
# _submit_exit records whether the exit was a profit-lock exit -- feeds the
# re-entry cooldown bypass (2026-08-28, see _reentry_cooldown_reason in
# test_incident_regressions.py section 18b).
# ---------------------------------------------------------------------------


def test_submit_exit_marks_a_profit_lock_exit_as_a_profit_exit(portfolio, config):
    broker = _FakeExecBroker(orderbook=None)
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 10_250.0, "고점 +3.00%에서 반락 -- +2.50% 확정 익절", [], {}, True, "",
        market=KOSPI, use_limit=True,
    )

    assert engine._last_exit_was_profit["005930"] is True


def test_submit_exit_marks_a_stop_loss_exit_as_not_a_profit_exit(portfolio, config):
    broker = _FakeExecBroker(orderbook=None)
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 9_870.0, "1.3% 손절 (-1.30%)", [], {}, True, "",
        market=KOSPI, use_limit=False,
    )

    assert engine._last_exit_was_profit["005930"] is False
