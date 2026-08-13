"""Broker adapters.

``BrokerBase`` is the narrow surface the trading loop uses. Two implementations
exist:

``DryRunBroker``    simulates an account and *prints* orders instead of sending
                    them. Used by ``--dry-run``, by every test, and whenever
                    credentials are missing.
``KiwoomBroker``    the real adapter, speaking Kiwoom's REST API. Its base URL
                    and its credentials both come from
                    :class:`settings.ModeDecision`, and on a paper run the live
                    app key was never loaded — so there is no code path that
                    reaches a live account without the triple confirmation.

Kiwoom throttles TR calls per second, so every request goes through a rate
limiter and a retry-with-backoff wrapper. ``requests`` is used directly rather
than a third-party wrapper because the loop needs to own that behaviour.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, TypeVar

import pandas as pd

from market.calendar import KST
from market.rules import KOSPI
from portfolio import LONG, SHORT
from settings import Credentials, ModeDecision, mask_text

logger = logging.getLogger("bot.broker")

T = TypeVar("T")

# Kiwoom REST paths and api-id codes.
# VERIFY these against https://openapi.kiwoom.com/ and the official examples at
# https://github.com/Kiwoom-Securities/Kiwoom-REST-API before live trading;
# they are overridable from config.yaml under `kiwoom.endpoints`.
DEFAULT_ENDPOINTS: dict[str, str] = {
    "token": "/oauth2/token",
    "order": "/api/dostk/ordr",
    "account": "/api/dostk/acnt",
    "chart": "/api/dostk/chart",
    "stock_info": "/api/dostk/stkinfo",
}

DEFAULT_API_IDS: dict[str, str] = {
    "buy": "kt10000",
    "sell": "kt10001",
    "cancel": "kt10003",
    "balance": "kt00018",
    "open_orders": "kt00007",
    "minute_chart": "ka10080",
    "daily_chart": "ka10081",
    "stock_info": "ka10001",
}


class BrokerError(RuntimeError):
    """Broker failure with any credential scrubbed from the message."""

    def __init__(self, message: str) -> None:
        super().__init__(mask_text(str(message)))


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    currency: str = "KRW"


@dataclass(frozen=True)
class StockInfo:
    code: str
    name: str = ""
    market: str = KOSPI
    tradable: bool = True
    previous_close: float = 0.0
    min_qty: float = 1.0


@dataclass
class OrderResult:
    code: str
    side: str
    qty: float
    order_type: str = "market"
    stop_price: float | None = None
    submitted: bool = False
    order_id: str = ""
    note: str = ""


# --------------------------------------------------------------------------
# Reliability helpers
# --------------------------------------------------------------------------


class RateLimiter:
    """Minimum spacing between API calls.

    Kiwoom rejects bursts, and a rejected order is indistinguishable from a
    dropped one from the loop's point of view, so requests are serialised
    rather than merely retried.
    """

    def __init__(self, calls_per_second: float = 4.0) -> None:
        self.min_interval = 1.0 / max(calls_per_second, 0.1)
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self, sleep: Callable[[float], None] = time.sleep) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                sleep(wait)
            self._last = time.monotonic()


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

    An API drop must not kill the loop. After the final attempt the error
    propagates (masked) so the caller can skip the cycle.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise BrokerError(
                    f"{description} failed after {max_retries} retries: {exc}"
                ) from None
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


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------


class BrokerBase:
    """Interface used by the trading loop."""

    dry_run: bool = True

    def get_account(self) -> AccountSnapshot:
        raise NotImplementedError

    def get_stock_info(self, code: str) -> StockInfo:
        raise NotImplementedError

    def open_order_codes(self) -> list[str]:
        raise NotImplementedError

    def get_chart(
        self, code: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame | None:
        return None

    def submit_order(
        self,
        code: str,
        side: str,
        qty: float,
        price: float | None = None,
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
    """Simulated broker. Never touches the network, never sends an order."""

    dry_run = True

    def __init__(
        self,
        starting_equity: float = 10_000_000.0,
        stocks: Mapping[str, StockInfo] | None = None,
        label: str = "DRY-RUN",
    ) -> None:
        self._equity = float(starting_equity)
        self._cash = float(starting_equity)
        self._stocks = dict(stocks or {})
        self.label = label
        self.submitted: list[OrderResult] = []
        self.cancelled = 0
        self.closed = 0

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=self._equity, cash=self._cash, buying_power=self._cash)

    def set_equity(self, equity: float, cash: float | None = None) -> None:
        self._equity = float(equity)
        self._cash = float(equity if cash is None else cash)

    def get_stock_info(self, code: str) -> StockInfo:
        return self._stocks.get(code, StockInfo(code=code))

    def open_order_codes(self) -> list[str]:
        return []

    def submit_order(
        self,
        code: str,
        side: str,
        qty: float,
        price: float | None = None,
        stop_price: float | None = None,
        is_exit: bool = False,
        note: str = "",
    ) -> OrderResult:
        result = OrderResult(
            code=code,
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
            code,
            qty,
            f"{stop_price:,.0f}" if stop_price is not None else "n/a",
            note or ("exit" if is_exit else "entry"),
        )
        return result

    def cancel_all_orders(self) -> int:
        self.cancelled += 1
        logger.info("[%s] would cancel all open orders — NOT sent", self.label)
        return 0

    def close_all_positions(self) -> int:
        self.closed += 1
        logger.info("[%s] would close all positions — NOT sent", self.label)
        return 0


# --------------------------------------------------------------------------
# Kiwoom
# --------------------------------------------------------------------------


class KiwoomBroker(BrokerBase):
    """Kiwoom REST adapter (mock or live, decided by ``ModeDecision``)."""

    dry_run = False

    def __init__(
        self,
        decision: ModeDecision,
        credentials: Credentials,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if not credentials.has_kiwoom:
            raise BrokerError(
                f"Kiwoom {decision.label} credentials are missing; set them in .env"
            )
        if credentials.loaded_for != decision.label:
            # Defensive: the credential set and the resolved mode must agree.
            raise BrokerError(
                f"credential set ({credentials.loaded_for}) does not match "
                f"resolved mode ({decision.label}); refusing to trade"
            )
        config = config or {}
        kiwoom_cfg = config.get("kiwoom") or {}
        api_cfg = config.get("api") or {}

        self.decision = decision
        self.credentials = credentials
        self.base_url = decision.endpoint.rstrip("/")
        self.endpoints = {**DEFAULT_ENDPOINTS, **(kiwoom_cfg.get("endpoints") or {})}
        self.api_ids = {**DEFAULT_API_IDS, **(kiwoom_cfg.get("api_ids") or {})}

        self.max_retries = int(api_cfg.get("max_retries", 5))
        self.base_seconds = float(api_cfg.get("backoff_base_seconds", 2.0))
        self.max_seconds = float(api_cfg.get("backoff_max_seconds", 60.0))
        self.timeout = float(api_cfg.get("timeout_seconds", 10.0))
        self.limiter = RateLimiter(float(api_cfg.get("calls_per_second", 4.0)))

        self._token: str = ""
        self._token_expires: datetime | None = None

    # -- auth --------------------------------------------------------------

    def _access_token(self) -> str:
        """Fetch and cache the OAuth access token, refreshing before expiry."""
        now = datetime.now(KST)
        if self._token and self._token_expires and now < self._token_expires:
            return self._token

        import requests

        url = self.base_url + self.endpoints["token"]
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.credentials.app_key.reveal(),
            "secretkey": self.credentials.secret_key.reveal(),
        }

        def _request():
            self.limiter.acquire()
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        data = with_retry(
            _request,
            max_retries=self.max_retries,
            base_seconds=self.base_seconds,
            max_seconds=self.max_seconds,
            description="kiwoom token",
        )
        token = data.get("token") or data.get("access_token") or ""
        if not token:
            raise BrokerError("token response contained no access token")
        self._token = token
        # Refresh a minute early rather than racing the expiry.
        self._token_expires = now + timedelta(seconds=int(data.get("expires_in", 3600)) - 60)
        logger.info("Kiwoom %s token acquired for %s", self.decision.label, self.base_url)
        return self._token

    # -- transport ---------------------------------------------------------

    def _call(
        self,
        endpoint_key: str,
        api_id_key: str,
        body: Mapping[str, Any],
        description: str,
    ) -> dict[str, Any]:
        import requests

        url = self.base_url + self.endpoints[endpoint_key]

        def _request():
            self.limiter.acquire()
            headers = {
                "Content-Type": "application/json;charset=UTF-8",
                "authorization": f"Bearer {self._access_token()}",
                "api-id": self.api_ids[api_id_key],
            }
            response = requests.post(url, json=dict(body), headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        return with_retry(
            _request,
            max_retries=self.max_retries,
            base_seconds=self.base_seconds,
            max_seconds=self.max_seconds,
            description=description,
        )

    # -- account -----------------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        data = self._call(
            "account",
            "balance",
            {"qry_tp": "1", "dmst_stex_tp": "KRX"},
            "get_account",
        )
        equity = _to_float(data.get("tot_evlt_amt") or data.get("prsm_dpst_aset_amt"))
        cash = _to_float(data.get("entr") or data.get("ord_alow_amt"))
        return AccountSnapshot(equity=equity, cash=cash, buying_power=cash)

    def get_stock_info(self, code: str) -> StockInfo:
        try:
            data = self._call("stock_info", "stock_info", {"stk_cd": code}, f"stock_info({code})")
        except BrokerError:
            # Unknown or unreachable: treat as not tradable rather than guessing.
            return StockInfo(code=code, tradable=False)
        return StockInfo(
            code=code,
            name=str(data.get("stk_nm") or ""),
            market=str(data.get("mrkt_tp") or KOSPI),
            tradable=str(data.get("trde_stop_yn") or "N").upper() != "Y",
            previous_close=abs(_to_float(data.get("base_pric") or data.get("prev_close"))),
        )

    def open_order_codes(self) -> list[str]:
        data = self._call("account", "open_orders", {"stex_tp": "0"}, "open_orders")
        rows = data.get("output") or data.get("oso") or []
        return [str(r.get("stk_cd") or "").strip() for r in rows if r.get("stk_cd")]

    def get_chart(
        self, code: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame | None:
        """Intraday/daily bars from Kiwoom — pykrx cannot serve minute data."""
        intraday = timeframe.endswith("Min")
        body: dict[str, Any] = {"stk_cd": code, "upd_stkpc_tp": "1"}
        if intraday:
            body["tic_scope"] = timeframe.replace("Min", "")
            key = "minute_chart"
        else:
            body["base_dt"] = end.strftime("%Y%m%d")
            key = "daily_chart"
        data = self._call("chart", key, body, f"chart({code},{timeframe})")
        rows = data.get("output") or next(
            (v for v in data.values() if isinstance(v, list) and v), []
        )
        if not rows:
            return None
        return _chart_frame(rows, intraday)

    # -- orders ------------------------------------------------------------

    def submit_order(
        self,
        code: str,
        side: str,
        qty: float,
        price: float | None = None,
        stop_price: float | None = None,
        is_exit: bool = False,
        note: str = "",
    ) -> OrderResult:
        api_id_key = "buy" if side == LONG else "sell"
        body = {
            "dmst_stex_tp": "KRX",
            "stk_cd": code,
            "ord_qty": str(int(qty)),
            "ord_uv": "" if price is None else str(int(price)),
            # "3" = market order; a limit price is only sent when one is given.
            "trde_tp": "3" if price is None else "0",
        }
        data = self._call("order", api_id_key, body, f"submit_order({code})")
        order_id = str(data.get("ord_no") or "")
        logger.info(
            "submitted %s %s x%s stop=%s id=%s",
            side,
            code,
            qty,
            f"{stop_price:,.0f}" if stop_price is not None else "n/a",
            order_id,
        )
        # Kiwoom has no bracket order, so the hard stop is enforced by the loop:
        # it checks the stop on every cycle and sends the exit itself.
        return OrderResult(
            code=code,
            side=side,
            qty=qty,
            order_type="market" if price is None else "limit",
            stop_price=stop_price,
            submitted=True,
            order_id=order_id,
            note=note,
        )

    def cancel_all_orders(self) -> int:
        count = 0
        for code in self.open_order_codes():
            try:
                self._call(
                    "order", "cancel", {"dmst_stex_tp": "KRX", "stk_cd": code}, f"cancel({code})"
                )
                count += 1
            except BrokerError as exc:
                logger.error("cancel failed for %s: %s", code, exc)
        logger.warning("cancelled %d open order(s)", count)
        return count

    def get_holdings(self) -> dict[str, float]:
        """Codes and quantities the account actually holds."""
        data = self._call(
            "account", "balance", {"qry_tp": "1", "dmst_stex_tp": "KRX"}, "get_holdings"
        )
        rows = data.get("output") or data.get("acnt_evlt_remn_indv_tot") or []
        holdings: dict[str, float] = {}
        for row in rows:
            code = str(row.get("stk_cd") or "").strip().lstrip("A")
            qty = _to_float(row.get("rmnd_qty") or row.get("hldg_qty"))
            if code and qty > 0:
                holdings[code] = qty
        return holdings

    def close_all_positions(self) -> int:
        """Flatten everything.

        Kiwoom has no flatten-everything endpoint, so holdings are read back
        from the account and sold one by one. Reading them from the broker
        rather than from local state matters here: the kill switch must close
        what the account *actually* holds, including anything the bot's own
        state file does not know about.
        """
        closed = 0
        for code, qty in self.get_holdings().items():
            try:
                self.submit_order(code, SHORT, qty, is_exit=True, note="kill switch flatten")
                closed += 1
            except BrokerError as exc:
                logger.error("flatten failed for %s: %s", code, exc)
        logger.warning("closed %d position(s)", closed)
        return closed


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", "").strip() or 0.0)
    except ValueError:
        return 0.0


def _chart_frame(rows: list[Mapping[str, Any]], intraday: bool) -> pd.DataFrame | None:
    """Normalise a Kiwoom chart payload into an OHLCV frame."""
    records = []
    for row in rows:
        stamp = row.get("cntr_tm") or row.get("dt") or row.get("stck_bsop_date")
        if not stamp:
            continue
        raw = "".join(ch for ch in str(stamp) if ch.isdigit())
        if intraday and len(raw) >= 14:
            fmt, raw = "%Y%m%d%H%M%S", raw[:14]
        elif intraday and len(raw) >= 12:
            fmt, raw = "%Y%m%d%H%M", raw[:12]
        elif len(raw) >= 8:
            fmt, raw = "%Y%m%d", raw[:8]
        else:
            continue
        try:
            when = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        records.append(
            {
                "timestamp": when.replace(tzinfo=KST),
                # Kiwoom returns signed prices on some charts; magnitude is the price.
                "open": abs(_to_float(row.get("open_pric"))),
                "high": abs(_to_float(row.get("high_pric"))),
                "low": abs(_to_float(row.get("low_pric"))),
                "close": abs(_to_float(row.get("cur_prc"))),
                "volume": abs(_to_float(row.get("trde_qty"))),
            }
        )
    if not records:
        return None
    frame = pd.DataFrame(records).set_index("timestamp").sort_index()
    return frame[frame["close"] > 0]


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
    starting_equity = float((config.get("account") or {}).get("starting_equity", 10_000_000.0))
    if force_dry_run:
        return DryRunBroker(starting_equity=starting_equity, label="DRY-RUN")
    if not credentials.has_kiwoom:
        logger.warning(
            "no Kiwoom %s credentials found in .env — falling back to the dry-run broker",
            decision.label,
        )
        return DryRunBroker(starting_equity=starting_equity, label="NO-CREDENTIALS")
    return KiwoomBroker(decision, credentials, config)
