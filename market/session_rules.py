"""Time-of-day rules for intraday and close-based strategies.

A strategy sees bars. It does not see the clock: a daily bar is stamped
``2026-08-19 00:00`` whether it is read at 09:05 or at 15:19, so a strategy
running on daily bars cannot tell the difference. Anything that depends on
*when in the session it is* therefore has to live outside the strategy, next to
the calendar -- which is what this module holds.

Two rules matter, and both are about the clock rather than the chart:

``entry_window``    when a new position may be opened at all. Day trading
                    stays out of the opening minutes, where the spread is
                    widest and the first bars have no channel behind them; a
                    close-based entry only happens in the last minutes of the
                    session, which is the whole point of it.

``force_exit``      when an open position is closed regardless of what the
                    strategy thinks. For a day trade this is what makes it a
                    day trade -- the position is flat before the closing
                    auction, where a market order has no continuous book to
                    reach. For an overnight position it is the planned exit
                    the next morning.

``force_exit`` deliberately fires *before* 15:20. A position still open when
the closing auction starts cannot be sold at a predictable price, and the next
chance is the following morning -- which is an overnight hold nobody asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Mapping

from market.calendar import CLOSING_AUCTION_START, KST, REGULAR_OPEN


def parse_clock(value: Any, field: str) -> time:
    """Parse ``"15:15"`` into a :class:`~datetime.time`."""
    if isinstance(value, time):
        return value
    text = str(value).strip()
    try:
        hour, minute = (int(part) for part in text.split(":"))
        return time(hour, minute)
    except (ValueError, TypeError):
        raise ValueError(
            f"{field}: expected a HH:MM time, got {value!r}"
        ) from None


@dataclass(frozen=True)
class SessionRules:
    """When a stock may be entered, and when it is closed regardless."""

    #: Earliest and latest moment a new position may be opened.
    entry_from: time | None = None
    entry_until: time | None = None
    #: Time of day at which an open position is closed unconditionally.
    force_exit_at: time | None = None
    #: True when force_exit_at refers to the morning *after* entry rather than
    #: the same session. Without this an overnight position would be closed
    #: seconds after it was opened, the exit time having already passed.
    hold_overnight: bool = False

    @classmethod
    def from_config(cls, asset_cfg: Mapping[str, Any]) -> "SessionRules":
        window = asset_cfg.get("entry_window") or []
        entry_from = entry_until = None
        if window:
            if len(window) != 2:
                raise ValueError(
                    f"entry_window must be [from, until], got {window!r}"
                )
            entry_from = parse_clock(window[0], "entry_window[0]")
            entry_until = parse_clock(window[1], "entry_window[1]")
        raw_exit = asset_cfg.get("force_exit_at")
        force_exit_at = parse_clock(raw_exit, "force_exit_at") if raw_exit else None
        rules = cls(
            entry_from=entry_from,
            entry_until=entry_until,
            force_exit_at=force_exit_at,
            hold_overnight=bool(asset_cfg.get("hold_overnight", False)),
        )
        rules.validate()
        return rules

    def validate(self) -> None:
        if self.entry_from and self.entry_until and self.entry_from >= self.entry_until:
            raise ValueError(
                f"entry_window opens at {self.entry_from} and closes at "
                f"{self.entry_until}, which is never open"
            )
        if self.force_exit_at and not self.hold_overnight:
            if self.force_exit_at >= CLOSING_AUCTION_START:
                raise ValueError(
                    f"force_exit_at {self.force_exit_at} is inside the closing "
                    f"auction ({CLOSING_AUCTION_START}); a market order there has "
                    "no continuous book, so the position would carry overnight"
                )
            if self.entry_until and self.entry_until >= self.force_exit_at:
                raise ValueError(
                    f"entry_window closes at {self.entry_until} but positions are "
                    f"flattened at {self.force_exit_at}; entries after that point "
                    "would be closed immediately"
                )
        if self.hold_overnight and self.force_exit_at:
            if self.force_exit_at < REGULAR_OPEN:
                raise ValueError(
                    f"force_exit_at {self.force_exit_at} is before the open "
                    f"({REGULAR_OPEN}); there is no market to sell into"
                )

    # -- queries -----------------------------------------------------------

    def entry_allowed(self, moment: datetime) -> tuple[bool, str]:
        """May a new position be opened right now?"""
        clock = _kst(moment).time()
        if self.entry_from and clock < self.entry_from:
            return False, f"entries open at {self.entry_from:%H:%M}"
        if self.entry_until and clock > self.entry_until:
            return False, f"entries closed at {self.entry_until:%H:%M}"
        return True, ""

    def exit_due(self, moment: datetime, entry_day: date | None) -> tuple[bool, str]:
        """Must an open position be closed right now, whatever the signal?

        *entry_day* is the session the position was opened in. For an
        overnight hold it is the difference between "the exit time has come"
        and "the exit time is later today, and this position is minutes old".
        """
        if self.force_exit_at is None:
            return False, ""
        local = _kst(moment)
        if local.time() < self.force_exit_at:
            return False, ""
        if self.hold_overnight:
            if entry_day is None or local.date() <= entry_day:
                return False, ""
            return True, (
                f"planned exit at {self.force_exit_at:%H:%M} the session after entry "
                f"({entry_day.isoformat()})"
            )
        return True, f"day-trade flat-out at {self.force_exit_at:%H:%M}"


def _kst(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=KST)
    return moment.astimezone(KST)
