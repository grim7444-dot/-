"""KRX session calendar - business days, session phases and auction safety.

The load-bearing rule here is that a market order must never be sent into a
call auction, where there is no continuous order book and the execution price
is not predictable.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from market.calendar import KST, KrxCalendar, SessionPhase


@pytest.fixture
def calendar():
    # 2025-01-01 (New Year) and 2025-03-03 as stand-ins for public holidays.
    return KrxCalendar(holidays=frozenset({date(2025, 1, 1), date(2025, 3, 3)}))


def at(y, m, d, hh, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=KST)


# --------------------------------------------------------------------------
# Business days
# --------------------------------------------------------------------------


def test_weekdays_are_business_days(calendar):
    assert calendar.is_business_day(date(2025, 1, 2)) is True   # Thursday


@pytest.mark.parametrize("day", [date(2025, 1, 4), date(2025, 1, 5)])
def test_weekends_are_closed(calendar, day):
    assert calendar.is_business_day(day) is False


def test_configured_holidays_are_closed(calendar):
    assert calendar.is_business_day(date(2025, 1, 1)) is False
    assert calendar.is_business_day(date(2025, 3, 3)) is False


def test_next_business_day_skips_the_weekend(calendar):
    # Friday 2025-01-03 -> Monday 2025-01-06
    assert calendar.next_business_day(date(2025, 1, 3)) == date(2025, 1, 6)


def test_next_business_day_skips_a_holiday(calendar):
    # Tuesday 2025-02-28 is a Friday; check the 3 March holiday is skipped.
    assert calendar.next_business_day(date(2025, 2, 28)) == date(2025, 3, 4)


def test_previous_business_day_skips_backwards(calendar):
    assert calendar.previous_business_day(date(2025, 1, 6)) == date(2025, 1, 3)


def test_business_days_excludes_weekends_and_holidays(calendar):
    days = calendar.business_days(date(2025, 1, 1), date(2025, 1, 7))
    assert date(2025, 1, 1) not in days   # holiday
    assert date(2025, 1, 4) not in days   # Saturday
    assert date(2025, 1, 2) in days
    assert len(days) == 4                 # 2,3,6,7


def test_missing_holiday_list_is_flagged_not_silently_assumed():
    """Without a holiday list we must know we only excluded weekends."""
    bare = KrxCalendar.from_config({})
    assert bare.holidays_unknown is True
    assert bare.is_business_day(date(2025, 1, 1)) is True  # a holiday it cannot know about


def test_holidays_load_from_config():
    cal = KrxCalendar.from_config({"krx": {"holidays": ["2025-01-01", "2025-05-05"]}})
    assert cal.holidays_unknown is False
    assert cal.is_business_day(date(2025, 5, 5)) is False


def test_unparseable_holiday_entries_are_ignored():
    cal = KrxCalendar.from_config({"krx": {"holidays": ["2025-01-01", "not-a-date"]}})
    assert cal.is_business_day(date(2025, 1, 1)) is False


# --------------------------------------------------------------------------
# Session phases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moment,expected",
    [
        (at(2025, 1, 2, 7, 59), SessionPhase.CLOSED),
        (at(2025, 1, 2, 8, 0), SessionPhase.NXT_PREMARKET),
        (at(2025, 1, 2, 8, 29), SessionPhase.NXT_PREMARKET),
        (at(2025, 1, 2, 8, 30), SessionPhase.OPENING_AUCTION),
        (at(2025, 1, 2, 8, 59), SessionPhase.OPENING_AUCTION),
        (at(2025, 1, 2, 9, 0), SessionPhase.CONTINUOUS),
        (at(2025, 1, 2, 12, 0), SessionPhase.CONTINUOUS),
        (at(2025, 1, 2, 15, 19), SessionPhase.CONTINUOUS),
        (at(2025, 1, 2, 15, 20), SessionPhase.CLOSING_AUCTION),
        (at(2025, 1, 2, 15, 29), SessionPhase.CLOSING_AUCTION),
        (at(2025, 1, 2, 15, 30), SessionPhase.CLOSED),
        (at(2025, 1, 2, 18, 0), SessionPhase.CLOSED),
    ],
)
def test_session_phase_boundaries(calendar, moment, expected):
    assert calendar.phase(moment) is expected


def test_weekend_is_closed_all_day(calendar):
    assert calendar.phase(at(2025, 1, 4, 11, 0)) is SessionPhase.CLOSED


def test_holiday_is_closed_all_day(calendar):
    assert calendar.phase(at(2025, 1, 1, 11, 0)) is SessionPhase.CLOSED


def test_is_open_is_continuous_trading_only(calendar):
    assert calendar.is_open(at(2025, 1, 2, 11, 0)) is True
    assert calendar.is_open(at(2025, 1, 2, 8, 45)) is False
    assert calendar.is_open(at(2025, 1, 2, 15, 25)) is False


def test_in_session_includes_the_auctions(calendar):
    assert calendar.in_session(at(2025, 1, 2, 8, 45)) is True
    assert calendar.in_session(at(2025, 1, 2, 15, 25)) is True
    assert calendar.in_session(at(2025, 1, 2, 18, 0)) is False


# --------------------------------------------------------------------------
# Market-order safety
# --------------------------------------------------------------------------


def test_market_orders_allowed_during_continuous_trading(calendar):
    allowed, reason = calendar.can_place_market_order(at(2025, 1, 2, 11, 0))
    assert allowed is True
    assert reason == ""


@pytest.mark.parametrize(
    "moment,fragment",
    [
        (at(2025, 1, 2, 8, 45), "opening call auction"),
        (at(2025, 1, 2, 15, 25), "closing call auction"),
        (at(2025, 1, 2, 18, 0), "market closed"),
        (at(2025, 1, 4, 11, 0), "market closed"),
    ],
)
def test_market_orders_refused_outside_continuous_trading(calendar, moment, fragment):
    allowed, reason = calendar.can_place_market_order(moment)
    assert allowed is False
    assert fragment in reason


# --------------------------------------------------------------------------
# Next open
# --------------------------------------------------------------------------


def test_next_open_is_today_when_called_before_the_bell(calendar):
    assert calendar.next_open(at(2025, 1, 2, 7, 0)) == at(2025, 1, 2, 9, 0)


def test_next_open_rolls_to_the_next_business_day_after_close(calendar):
    assert calendar.next_open(at(2025, 1, 2, 16, 0)) == at(2025, 1, 3, 9, 0)


def test_next_open_skips_the_weekend(calendar):
    assert calendar.next_open(at(2025, 1, 3, 16, 0)) == at(2025, 1, 6, 9, 0)


def test_seconds_until_open_is_zero_while_trading(calendar):
    assert calendar.seconds_until_open(at(2025, 1, 2, 11, 0)) == 0.0


def test_seconds_until_open_is_positive_when_closed(calendar):
    assert calendar.seconds_until_open(at(2025, 1, 2, 7, 0)) == pytest.approx(2 * 3600)


def test_naive_datetimes_are_treated_as_kst(calendar):
    """A naive timestamp must not silently be read as UTC - that is 9 hours off."""
    assert calendar.phase(datetime(2025, 1, 2, 11, 0)) is SessionPhase.CONTINUOUS
