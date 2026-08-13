"""Broker adapters.

``BrokerBase`` is the narrow surface the trading loop uses.  Two
implementations exist:

``DryRunBroker``   simulates an account and *prints* orders instead of sending
                   them.  Used by ``--dry-run``, by every test, and whenever
                   credentials are missing.
``AlpacaBroker``   the real adapter.  Its base URL comes from
                   :class:`settings.ModeDecision` and nowhere else, so there is
                   no code path that reaches the live endpoint without the
                   triple confirmation having passed.

The ``alpaca`` SDK is imported lazily inside ``AlpacaBroker`` so the module —
and the whole test suite — imports fine without it.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, TypeVar

from portfolio import LONG, SHORT
from settings import Credentials, ModeDecision, mask_text

logger = logging.getLogger("bot.broker")

T = TypeVar("T")


class BrokerError(RuntimeError):
    """Broker failure with any credential scrubbed from the message."""

    def __init__(self, message: str) -> None:
        super().__init__(mask_text(str(message)))


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"


@dataclass(frozen=True)
class AssetInfo:
    symbol: str
    tradable: bool = True
    fractionable: bool = False
    min_qty: float = 1.0


@dataclass
class OrderResult:
    symbol: str
    side: str
    qty: float
    order_type: str = "market"
    stop_price: float | None = None
    submitted: bool = False
    order_id: str = ""
    note: str = ""
    filled_price: float | None = None


def with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 5,
    base_seconds: float = 2.0,
    max_seconds: float = 60.0,
    description: str = "broker call",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run *fn*, retrying transient API failures with exponential backoff.

    Rule: an API drop must not kill the loop.  After the final attempt the
    error propagates (masked) so the caller can decide to skip the cycle.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise BrokerError(f"{description} failed after {max_retries} retries: {exc}") from None
            delay = min(base_seconds * (2 ** (attempt - 1)), max_seconds)
            delay += random.uniform(0, delay * 0.1)  # jitter
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                description,
                attempt,
                max_retries,
                mask_text(str(exc)),
                delay,
            )
            sleep(delay)


class BrokerBase:
    """Interface used by the trading loop."""

    #: True when orders are only simulated.
    dry_run: bool = True

    def get_account(self) -> AccountSnapshot:
        raise NotImplementedError

    def is_market_open(self) -> bool:
        raise NotImplementedError

    def get_asset(self, symbol: str) -> AssetInfo:
        raise NotImplementedError

    def open_order_symbols(self) -> list[str]:
        raise NotImplementedError

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float | None = None,
        is_exit: bool = False,
        note: str = "",
    ) -> OrderResult:
        raise NotImplementedError

    def cancel_all_orders(self) -> int:
        raise NotImplementedError

    def close_all_positions(self) -> int:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------


class DryRunBroker(BrokerBase):
    """Simulated broker.  Never touches the network, never sends an order."""

    dry_run = True

    def __init__(
        self,
        starting_equity: float = 100_000.0,
        market_open: bool = True,
        assets: Mapping[str, AssetInfo] | None = None,
        label: str = "DRY-RUN",
    ) -> None:
        self._equity = float(starting_equity)
        self._cash = float(starting_equity)
        self._market_open = market_open
        self._assets = dict(assets or {})
        self.label = label
        self.submitted: list[OrderResult] = []
        self._cancelled = 0
        self._closed = 0

    # -- account -----------------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=self._equity, cash=self._cash, buying_power=self._cash)

    def set_equity(self, equity: float, cash: float | None = None) -> None:
        self._equity = float(equity)
        self._cash = float(equity if cash is None else cash)

    def is_market_open(self) -> bool:
        return self._market_open

    def set_market_open(self, is_open: bool) -> None:
        self._market_open = is_open

    def get_asset(self, symbol: str) -> AssetInfo:
        return self._assets.get(symbol, AssetInfo(symbol=symbol))

    def open_order_symbols(self) -> list[str]:
        return [o.symbol for o in self.submitted if o.submitted is False and o.note == "working"]

    # -- orders ------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float | None = None,
        is_exit: bool = False,
        note: str = "",
    ) -> OrderResult:
        result = OrderResult(
            symbol=symbol,
            side=side,
            qty=qty,
            stop_price=stop_price,
            submitted=False,
            note=note or "simulated",
        )
        self.submitted.append(result)
        logger.info(
            "[%s] would submit %s %s x%s stop=%s (%s) — NOT sent",
            self.label,
            side,
            symbol,
            qty,
            f"{stop_price:.4f}" if stop_price is not None else "n/a",
            note or ("exit" if is_exit else "entry"),
        )
        return result

    def cancel_all_orders(self) -> int:
        self._cancelled += 1
        logger.info("[%s] would cancel all open orders — NOT sent", self.label)
        return 0

    def close_all_positions(self) -> int:
        self._closed += 1
        logger.info("[%s] would close all positions — NOT sent", self.label)
        return 0


# --------------------------------------------------------------------------
# Alpaca
# --------------------------------------------------------------------------


class AlpacaBroker(BrokerBase):
    """Real Alpaca adapter (paper or live, decided by ``ModeDecision``)."""

    dry_run = False

    def __init__(
        self,
        decision: ModeDecision,
        credentials: Credentials,
        api_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        if not credentials.has_alpaca:
            raise BrokerError("Alpaca credentials are missing; set them in .env")
        self.decision = decision
        self.credentials = credentials
        api_cfg = api_cfg or {}
        self.max_retries = int(api_cfg.get("max_retries", 5))
        self.base_seconds = float(api_cfg.get("backoff_base_seconds", 2.0))
        self.max_seconds = float(api_cfg.get("backoff_max_seconds", 60.0))
        self._client: Any | None = None

    # -- client ------------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            # Lazy import so the SDK stays an optional dependency for tests.
            from alpaca.trading.client import TradingClient  # type: ignore

            # `paper` is derived from the gate in settings.resolve_mode; this is
            # the only place the flag is consumed.
            self._client = TradingClient(
                api_key=self.credentials.api_key.reveal(),
                secret_key=self.credentials.secret_key.reveal(),
                paper=self.decision.paper,
            )
            logger.info("Alpaca client connected to %s", self.decision.endpoint)
        return self._client

    def _call(self, fn: Callable[[], T], description: str) -> T:
        return with_retry(
            fn,
            max_retries=self.max_retries,
            base_seconds=self.base_seconds,
            max_seconds=self.max_seconds,
            description=description,
        )

    # -- account -----------------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        account = self._call(lambda: self.client.get_account(), "get_account")
        return AccountSnapshot(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            currency=getattr(account, "currency", "USD") or "USD",
        )

    def is_market_open(self) -> bool:
        clock = self._call(lambda: self.client.get_clock(), "get_clock")
        return bool(clock.is_open)

    def next_market_open(self):
        clock = self._call(lambda: self.client.get_clock(), "get_clock")
        return getattr(clock, "next_open", None)

    def get_asset(self, symbol: str) -> AssetInfo:
        try:
            asset = self._call(lambda: self.client.get_asset(symbol), f"get_asset({symbol})")
        except BrokerError:
            # Unknown symbol: treat as not tradable rather than guessing.
            return AssetInfo(symbol=symbol, tradable=False)
        return AssetInfo(
            symbol=symbol,
            tradable=bool(getattr(asset, "tradable", False)),
            fractionable=bool(getattr(asset, "fractionable", False)),
            min_qty=float(getattr(asset, "min_order_size", None) or 1.0),
        )

    def open_order_symbols(self) -> list[str]:
        from alpaca.trading.requests import GetOrdersRequest  # type: ignore
        from alpaca.trading.enums import QueryOrderStatus  # type: ignore

        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = self._call(lambda: self.client.get_orders(request), "get_orders")
        return [o.symbol for o in orders or []]

    def get_positions(self) -> dict[str, Any]:
        positions = self._call(lambda: self.client.get_all_positions(), "get_all_positions")
        return {p.symbol: p for p in positions or []}

    # -- orders ------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float | None = None,
        is_exit: bool = False,
        note: str = "",
    ) -> OrderResult:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce  # type: ignore
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest  # type: ignore

        order_side = OrderSide.BUY if side == LONG else OrderSide.SELL
        kwargs: dict[str, Any] = {
            "symbol": symbol,
            "qty": qty,
            "side": order_side,
            "time_in_force": TimeInForce.GTC if "/" in symbol else TimeInForce.DAY,
        }
        if stop_price is not None and not is_exit:
            # Rule: the hard stop is attached to the entry, never bolted on
            # afterwards, so an order cannot exist without its stop.
            kwargs["order_class"] = OrderClass.OTO
            kwargs["stop_loss"] = StopLossRequest(stop_price=round(stop_price, 2))

        request = MarketOrderRequest(**kwargs)
        order = self._call(lambda: self.client.submit_order(request), f"submit_order({symbol})")
        logger.info(
            "submitted %s %s x%s stop=%s id=%s",
            side,
            symbol,
            qty,
            f"{stop_price:.4f}" if stop_price is not None else "n/a",
            getattr(order, "id", ""),
        )
        return OrderResult(
            symbol=symbol,
            side=side,
            qty=qty,
            stop_price=stop_price,
            submitted=True,
            order_id=str(getattr(order, "id", "")),
            note=note,
        )

    def cancel_all_orders(self) -> int:
        responses = self._call(lambda: self.client.cancel_orders(), "cancel_orders")
        count = len(responses or [])
        logger.warning("cancelled %d open order(s)", count)
        return count

    def close_all_positions(self) -> int:
        responses = self._call(
            lambda: self.client.close_all_positions(cancel_orders=True), "close_all_positions"
        )
        count = len(responses or [])
        logger.warning("closed %d position(s)", count)
        return count


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def build_broker(
    decision: ModeDecision,
    credentials: Credentials,
    config: Mapping[str, Any],
    force_dry_run: bool = False,
) -> BrokerBase:
    """Pick the broker for this run.

    Without credentials, or with ``--dry-run``, we fall back to the simulated
    broker: the bot stays usable end to end but cannot place an order.
    """
    starting_equity = float((config.get("account") or {}).get("starting_equity", 100_000.0))
    if force_dry_run:
        return DryRunBroker(starting_equity=starting_equity, label="DRY-RUN")
    if not credentials.has_alpaca:
        logger.warning(
            "no Alpaca credentials found in .env — falling back to the dry-run broker"
        )
        return DryRunBroker(starting_equity=starting_equity, label="NO-CREDENTIALS")
    return AlpacaBroker(decision, credentials, config.get("api"))
