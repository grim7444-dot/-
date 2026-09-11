"""KRX session calendar.

Korean equities trade one continuous session, which makes scheduling simpler
than the US bot's split of 24/7 crypto against a regular session - but the
session has *phases*, and one of them matters for order safety:

    08:30-09:00   opening call auction   (주문 접수, 단일가 결정)
    09:00-15:30   continuous trading     (정규장)
    15:20-15:30   closing call auction   (종가 단일가)

During a call auction there is no continuous order book, so a market order's
execution price is not predictable. The bot refuses to send market orders in
those windows and waits for continuous trading.

Holidays cannot be derived from a rule - Korea's calendar includes lunar
holidays, substitute holidays and ad-hoc closures. They are supplied through
``config.yaml`` and can be refreshed from pykrx when a network is available.
With neither source the calendar degrades to weekends-only and says so, rather
than silently treating a holiday as a trading day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

logger = logging.getLogger("bot.calendar")

KST = ZoneInfo("Asia/Seoul")

NXT_PREMARKET_START = time(8, 0)   # NXT (NextTrade ATS) opens for pre-market
NXT_PREMARKET_END = time(8, 30)    # KRX opening auction starts; NXT bias is set
OPENING_AUCTION_START = time(8, 30)
REGULAR_OPEN = time(9, 0)
CLOSING_AUCTION_START = time(15, 20)
REGULAR_CLOSE = time(15, 30)


class SessionPhase(str, Enum):
    CLOSED = "CLOSED"
    NXT_PREMARKET = "NXT_PREMARKET"
    OPENING_AUCTION = "OPENING_AUCTION"
    CONTINUOUS = "CONTINUOUS"
    CLOSING_AUCTION = "CLOSING_AUCTION"

    @property
    def tradable(self) -> bool:
        """Only continuous trading accepts a market order safely."""
        return self is SessionPhase.CONTINUOUS


@dataclass
class KrxCalendar:
    """Session phases and business days for KRX."""

    holidays: frozenset[date] = frozenset()
    #: True when no holiday list was supplied, so only weekends are excluded.
    holidays_unknown: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "KrxCalendar":
        raw = (config.get("krx") or {}).get("holidays") or []
        parsed: set[date] = set()
        for item in raw:
            if isinstance(item, date):
                parsed.add(item)
            else:
                try:
                    parsed.add(date.fromisoformat(str(item)))
                except ValueError:
                    logger.warning("ignoring unparseable holiday entry: %r", item)
        if not parsed:
            logger.warning(
                "no KRX holiday list configured - only weekends will be treated as "
                "closed. Run `python main.py profile --refresh-calendar` with network "
                "access, or fill krx.holidays in config.yaml."
            )
        return cls(holidays=frozenset(parsed), holidays_unknown=not parsed)

    # -- days --------------------------------------------------------------

    def is_business_day(self, day: date) -> bool:
        if day.weekday() >= 5:
            return False
        return day not in self.holidays

    def previous_business_day(self, day: date) -> date:
        cursor = day - timedelta(days=1)
        for _ in range(30):
            if self.is_business_day(cursor):
                return cursor
            cursor -= timedelta(days=1)
        return cursor

    def next_business_day(self, day: date) -> date:
        cursor = day + timedelta(days=1)
        for _ in range(30):
            if self.is_business_day(cursor):
                return cursor
            cursor += timedelta(days=1)
        return cursor

    def business_days(self, start: date, end: date) -> list[date]:
        out: list[date] = []
        cursor = start
        while cursor <= end:
            if self.is_business_day(cursor):
                out.append(cursor)
            cursor += timedelta(days=1)
        return out

    # -- intraday ----------------------------------------------------------

    def _to_kst(self, moment: datetime | None) -> datetime:
        moment = moment or datetime.now(KST)
        if moment.tzinfo is None:
            return moment.replace(tzinfo=KST)
        return moment.astimezone(KST)

    def phase(self, moment: datetime | None = None) -> SessionPhase:
        local = self._to_kst(moment)
        if not self.is_business_day(local.date()):
            return SessionPhase.CLOSED
        clock = local.time()
        if NXT_PREMARKET_START <= clock < NXT_PREMARKET_END:
            return SessionPhase.NXT_PREMARKET
        if OPENING_AUCTION_START <= clock < REGULAR_OPEN:
            return SessionPhase.OPENING_AUCTION
        if CLOSING_AUCTION_START <= clock < REGULAR_CLOSE:
            return SessionPhase.CLOSING_AUCTION
        if REGULAR_OPEN <= clock < CLOSING_AUCTION_START:
            return SessionPhase.CONTINUOUS
        return SessionPhase.CLOSED

    def is_nxt_premarket(self, moment: datetime | None = None) -> bool:
        return self.phase(moment) is SessionPhase.NXT_PREMARKET

    def is_open(self, moment: datetime | None = None) -> bool:
        """True during continuous trading only."""
        return self.phase(moment) is SessionPhase.CONTINUOUS

    def in_session(self, moment: datetime | None = None) -> bool:
        """True during continuous trading or either auction (not NXT pre-market)."""
        return self.phase(moment) not in (SessionPhase.CLOSED, SessionPhase.NXT_PREMARKET)

    def can_place_market_order(self, moment: datetime | None = None) -> tuple[bool, str]:
        """Market orders are only safe during continuous trading."""
        phase = self.phase(moment)
        if phase is SessionPhase.CONTINUOUS:
            return True, ""
        if phase is SessionPhase.NXT_PREMARKET:
            return False, "NXT pre-market (08:00-08:30): bias scan only, no orders"
        if phase is SessionPhase.OPENING_AUCTION:
            return False, "opening call auction (08:30-09:00): no continuous order book"
        if phase is SessionPhase.CLOSING_AUCTION:
            return False, "closing call auction (15:20-15:30): no continuous order book"
        return False, "market closed (KRX trades 09:00-15:30 KST on business days)"

    def next_nxt_start(self, moment: datetime | None = None) -> datetime:
        """The next NXT pre-market start after *moment*."""
        local = self._to_kst(moment)
        today_nxt = datetime.combine(local.date(), NXT_PREMARKET_START, tzinfo=KST)
        if self.is_business_day(local.date()) and local < today_nxt:
            return today_nxt
        return datetime.combine(self.next_business_day(local.date()), NXT_PREMARKET_START, tzinfo=KST)

    def next_open(self, moment: datetime | None = None) -> datetime:
        """The next continuous-trading start after *moment*."""
        local = self._to_kst(moment)
        today_open = datetime.combine(local.date(), REGULAR_OPEN, tzinfo=KST)
        if self.is_business_day(local.date()) and local < today_open:
            return today_open
        return datetime.combine(self.next_business_day(local.date()), REGULAR_OPEN, tzinfo=KST)

    def seconds_until_open(self, moment: datetime | None = None) -> float:
        local = self._to_kst(moment)
        if self.is_open(local):
            return 0.0
        return max(0.0, (self.next_open(local) - local).total_seconds())

    # -- maintenance -------------------------------------------------------

    def refresh_from_pykrx(self, start: date, end: date) -> set[date]:
        """Derive the holiday set from pykrx's business-day list.

        Returns the weekday dates in range that KRX did *not* trade. Requires
        network access; the caller writes the result into config.yaml.
        """
        # Lazy import: pykrx is only needed for this maintenance path.
        from data import _import_pykrx_stock

        stock = _import_pykrx_stock()

        traded = {
            datetime.strptime(d, "%Y%m%d").date()
            for d in stock.get_previous_business_days(
                fromdate=start.strftime("%Y%m%d"), todate=end.strftime("%Y%m%d")
            )
        }
        weekdays = {
            d
            for d in (start + timedelta(days=i) for i in range((end - start).days + 1))
            if d.weekday() < 5
        }
        return weekdays - traded
