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
    def __init__(
        self, orderbook=None, reject_with=None, holdings=None, holdings_error=None,
        dry_run=False,
    ):
        self._orderbook = orderbook
        self.orders: list[dict] = []
        #: If set, submit_order raises this BrokerError instead of filling.
        self._reject_with = reject_with
        #: dict[str, Holding] returned by get_holdings(), or an exception
        #: instance/class to raise instead.
        self._holdings = holdings or {}
        self._holdings_error = holdings_error
        self.dry_run = dry_run

    def get_stock_info(self, code):
        from broker import StockInfo

        return StockInfo(code=code, tradable=True)

    def get_orderbook(self, code):
        return self._orderbook

    def get_holdings(self):
        if self._holdings_error is not None:
            raise self._holdings_error
        return self._holdings

    def submit_order(self, **kwargs):
        from broker import OrderResult

        self.orders.append(kwargs)
        if self._reject_with is not None:
            raise self._reject_with
        return OrderResult(
            code=kwargs["code"], side=kwargs["side"], qty=kwargs["qty"], submitted=True, order_id="1",
        )


class _FakeExecNotifier:
    def __init__(self):
        self.sent: list[str] = []

    def send(self, text, *a, **k):
        self.sent.append(text)

    def alert_exit(self, *a, **k):
        pass


def _exec_engine(
    portfolio, config, broker, use_limit_orders=True, limit_buffer_ticks=2,
    stop_limit_buffer_ticks=5,
):
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
    engine._stop_limit_buffer_ticks = stop_limit_buffer_ticks
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
        market=KOSPI, urgent=False,
    )

    assert len(broker.orders) == 1
    assert broker.orders[0]["price"] == 9_980.0  # sell: best_bid - 2 ticks


def test_submit_exit_uses_a_wider_limit_price_for_an_urgent_exit(portfolio, config):
    """The hard-stop / forced-time-exit path (2026-08-28, user request: 시장가로
    하지 말고 -- don't use plain market orders here either). It crosses the
    book by stop_limit_buffer_ticks (5, wider than the normal 2) instead of
    going in with no price cap at all."""
    book = _FakeOrderBook(best_bid=10_000.0, best_ask=10_010.0)
    broker = _FakeExecBroker(orderbook=book)
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 9_800.0, "stop hit", [], {}, True, "",
        market=KOSPI, urgent=True,
    )

    assert len(broker.orders) == 1
    assert broker.orders[0]["price"] == 9_950.0  # sell: best_bid - 5 ticks


def test_submit_exit_falls_back_to_market_with_no_orderbook_even_when_urgent(portfolio, config):
    """The one case nothing can bound: no quote data at all means a plain
    market order regardless of urgency -- same fail-open as the take-profit
    path, never a reason to refuse an otherwise-valid exit."""
    broker = _FakeExecBroker(orderbook=None)
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 9_800.0, "stop hit", [], {}, True, "",
        market=KOSPI, urgent=True,
    )

    assert broker.orders[0]["price"] is None


def test_submit_exit_falls_back_to_market_with_no_orderbook_even_when_not_urgent(portfolio, config):
    broker = _FakeExecBroker(orderbook=None)
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 10_050.0, "확정 익절", [], {}, True, "",
        market=KOSPI, urgent=False,
    )

    assert broker.orders[0]["price"] is None


def test_submit_exit_respects_the_global_use_limit_orders_toggle(portfolio, config):
    book = _FakeOrderBook(best_bid=10_000.0, best_ask=10_010.0)
    broker = _FakeExecBroker(orderbook=book)
    engine = _exec_engine(portfolio, config, broker, use_limit_orders=False)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 10_050.0, "확정 익절", [], {}, True, "",
        market=KOSPI, urgent=False,
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
        market=KOSPI, urgent=False,
    )

    assert engine._last_exit_was_profit["005930"] is True


def test_submit_exit_marks_a_stop_loss_exit_as_not_a_profit_exit(portfolio, config):
    broker = _FakeExecBroker(orderbook=None)
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 9_870.0, "1.3% 손절 (-1.30%)", [], {}, True, "",
        market=KOSPI, urgent=True,
    )

    assert engine._last_exit_was_profit["005930"] is False


# ---------------------------------------------------------------------------
# A rejected exit reconciles against the broker's actual holdings instead of
# leaving a stale position to retry forever (2026-08-28 live incident: 122640
# 예스티 stayed in state.json after Kiwoom's real sellable qty was already 0,
# and every cycle re-tried the same rejected sell with no way out).
# ---------------------------------------------------------------------------


def test_rejected_exit_clears_a_position_the_broker_confirms_is_gone(portfolio, config):
    from broker import BrokerError

    broker = _FakeExecBroker(
        reject_with=BrokerError(
            "order for 005930 was not accepted [return_code=20] "
            "[2000](800033:매도가능수량이 부족합니다. 0주 매도가능)"
        ),
        holdings={},  # broker reports nothing held for this code
    )
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 30_350.0, "stop hit", [], {}, True, "",
        market=KOSPI, urgent=True,
    )

    assert portfolio.get("005930") is None  # stale local position cleared
    assert engine._tg_notifier.sent  # user was told


def test_rejected_exit_leaves_the_position_when_broker_still_holds_it(portfolio, config):
    from broker import BrokerError, Holding

    broker = _FakeExecBroker(
        reject_with=BrokerError("temporary rejection"),
        holdings={"005930": Holding(code="005930", qty=10.0)},
    )
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 30_350.0, "stop hit", [], {}, True, "",
        market=KOSPI, urgent=True,
    )

    assert portfolio.get("005930") is not None  # left alone for the next cycle to retry


def test_rejected_exit_does_nothing_when_the_holdings_check_also_fails(portfolio, config):
    from broker import BrokerError

    broker = _FakeExecBroker(
        reject_with=BrokerError("temporary rejection"),
        holdings_error=BrokerError("holdings unavailable too"),
    )
    engine = _exec_engine(portfolio, config, broker)
    position = _open_exec_position(portfolio)

    engine._submit_exit(
        "005930", position, 30_350.0, "stop hit", [], {}, True, "",
        market=KOSPI, urgent=True,
    )

    assert portfolio.get("005930") is not None  # cannot confirm either way -- left alone


# ---------------------------------------------------------------------------
# _reconcile_holdings_with_broker -- catches a manual buy/sell the bot never
# saw once per cycle, instead of only when its own exit logic happens to try
# acting on the stale position and gets rejected (2026-08-31, user-reported:
# 내가 금호전기도 매수매도 했고 지금 현대약품도 매도 직접했는데 이걸 봇이
# 알고 있나? -- a position sold by hand while sitting quietly between its
# stop and its lock floor could go a long time before the bot ever noticed).
# ---------------------------------------------------------------------------


def test_reconcile_clears_a_position_sold_outside_the_bot(portfolio, config):
    broker = _FakeExecBroker(holdings={})  # broker confirms nothing held
    engine = _exec_engine(portfolio, config, broker)
    _open_exec_position(portfolio)

    engine._reconcile_holdings_with_broker()

    assert portfolio.get("005930") is None
    assert engine._tg_notifier.sent


def test_reconcile_leaves_a_position_the_broker_still_holds(portfolio, config):
    from broker import Holding

    broker = _FakeExecBroker(holdings={"005930": Holding(code="005930", qty=10.0)})
    engine = _exec_engine(portfolio, config, broker)
    _open_exec_position(portfolio)

    engine._reconcile_holdings_with_broker()

    assert portfolio.get("005930") is not None
    assert engine._tg_notifier.sent == []


def test_reconcile_does_nothing_in_dry_run_mode(portfolio, config):
    """DryRunBroker doesn't even implement get_holdings() -- this must never
    be called for a paper/dry-run session."""
    broker = _FakeExecBroker(dry_run=True, holdings_error=AttributeError("should not be called"))
    engine = _exec_engine(portfolio, config, broker)
    _open_exec_position(portfolio)

    engine._reconcile_holdings_with_broker()  # must not raise

    assert portfolio.get("005930") is not None


def test_reconcile_does_nothing_with_no_open_positions(portfolio, config):
    broker = _FakeExecBroker(holdings_error=AssertionError("should not be called"))
    engine = _exec_engine(portfolio, config, broker)

    engine._reconcile_holdings_with_broker()  # must not raise, must not call get_holdings


def test_reconcile_leaves_positions_alone_when_the_holdings_check_fails(portfolio, config):
    from broker import BrokerError

    broker = _FakeExecBroker(holdings_error=BrokerError("holdings unavailable"))
    engine = _exec_engine(portfolio, config, broker)
    _open_exec_position(portfolio)

    engine._reconcile_holdings_with_broker()

    assert portfolio.get("005930") is not None
