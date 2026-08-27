"""Daily 복기 (retrospective) and the Friday weekly rollup.

User request (2026-08-27, Korean): 끝나면 복기하고 항상 뭐가 잘못됐는지
보여주고 매주 금요일에 일주일치를 보여줘. Every evening report must always
show what went wrong (not just a neutral PnL number), and a wider weekly
view rolls up every Friday.
"""

from __future__ import annotations

from datetime import date

from risk.manager import RiskManager


def _trade(
    day: str,
    symbol: str = "005930",
    pnl: float = 0.0,
    exit_reason: str = "",
    fees: float = 0.0,
):
    from portfolio import TradeRecord

    return TradeRecord(
        timestamp=f"{day}T14:00:00+09:00",
        symbol=symbol,
        side="LONG",
        entry_price=10_000.0,
        exit_price=10_000.0 + pnl,
        pnl=pnl,
        qty=1,
        fees=fees,
        exit_reason=exit_reason,
    )


# ---------------------------------------------------------------------------
# _bucket_exit_reason: matched against the real strings main.py's
# _submit_exit() call sites actually use.
# ---------------------------------------------------------------------------


def test_bucket_recognizes_the_hard_atr_stop():
    from report import _bucket_exit_reason

    assert _bucket_exit_reason("stop hit") == "손절"


def test_bucket_recognizes_a_strategys_own_percent_stop():
    from report import _bucket_exit_reason

    assert _bucket_exit_reason("1.7% 손절 (-1.71%)") == "손절"


def test_bucket_recognizes_the_normal_lock_tier_exit():
    from report import _bucket_exit_reason

    assert _bucket_exit_reason("고점 +2.50%에서 반락 -- 2.10% 확정 익절") == "확정 익절"


def test_bucket_recognizes_the_late_session_take_profit():
    from report import _bucket_exit_reason

    assert _bucket_exit_reason("장마감 15분 전, 고점 대비 -1.60% 정체 -- 조기 익절 (+0.80%)") \
        == "조기 익절 (마감 임박)"


def test_bucket_recognizes_a_time_based_force_exit():
    from report import _bucket_exit_reason

    assert _bucket_exit_reason("day-trade flat-out at 15:10") == "시간 청산"
    assert _bucket_exit_reason("planned exit at 09:05 the session after entry (2026-08-26)") \
        == "시간 청산"


def test_bucket_falls_back_to_기타_for_anything_unrecognized():
    from report import _bucket_exit_reason

    assert _bucket_exit_reason("") == "기타"
    assert _bucket_exit_reason("some unrelated text") == "기타"


# ---------------------------------------------------------------------------
# _retrospective_lines / build_evening_report: always show what went wrong
# ---------------------------------------------------------------------------


def test_retrospective_lists_every_losing_trade_with_its_reason(config):
    from report import _retrospective_lines

    rows = [
        _trade("2026-08-27", symbol="005930", pnl=-500.0, exit_reason="stop hit").to_row(),
        _trade("2026-08-27", symbol="082800", pnl=1_200.0, exit_reason="확정 익절").to_row(),
    ]
    lines = _retrospective_lines(rows, config)
    text = "\n".join(lines)
    assert "005930" in text
    assert "stop hit" in text
    assert "082800" not in text  # the winner has no business in the problem list


def test_retrospective_says_so_explicitly_when_there_are_no_losers(config):
    from report import _retrospective_lines

    rows = [_trade("2026-08-27", pnl=1_000.0, exit_reason="확정 익절").to_row()]
    text = "\n".join(_retrospective_lines(rows, config))
    assert "손실 거래 없음" in text


def test_retrospective_handles_a_day_with_no_trades_at_all(config):
    from report import _retrospective_lines

    text = "\n".join(_retrospective_lines([], config))
    assert "청산된 거래 없음" in text


def test_evening_report_always_includes_the_복기_section(portfolio, config):
    from portfolio import TradeRecord
    from report import build_evening_report

    today = date(2026, 8, 27)
    portfolio.trades.record(_trade(today.isoformat(), symbol="032820", pnl=-800.0, exit_reason="stop hit"))
    risk = RiskManager(config, portfolio)

    text = build_evening_report(config, portfolio, risk, "PAPER", day=today)
    assert "복기" in text
    assert "032820" in text
    assert "stop hit" in text


# ---------------------------------------------------------------------------
# build_weekly_report
# ---------------------------------------------------------------------------


def test_weekly_report_rolls_up_the_trailing_seven_days_only(portfolio, config):
    """A loss just outside the 7-day window must not leak into the totals
    or the 복기 loser list -- only the window's own trade counts."""
    from report import build_weekly_report

    week_end = date(2026, 8, 28)  # a Friday
    in_window = date(2026, 8, 24)   # 4 days before -- inside the 7-day window
    outside_window = date(2026, 8, 20)  # 8 days before -- outside it

    portfolio.trades.record(_trade(in_window.isoformat(), symbol="005930", pnl=500.0, exit_reason="확정 익절"))
    portfolio.trades.record(_trade(outside_window.isoformat(), symbol="999999", pnl=-9_999.0, exit_reason="stop hit"))
    risk = RiskManager(config, portfolio)

    text = build_weekly_report(config, portfolio, risk, "PAPER", week_end=week_end)
    assert "Trades closed : 1" in text
    assert "+500" in text
    assert "999999" not in text
    assert "-9,999" not in text
    assert "손실 거래 없음" in text  # the only in-window trade is a winner


def test_weekly_report_breaks_down_exit_reasons_by_type(portfolio, config):
    from report import build_weekly_report

    week_end = date(2026, 8, 28)
    portfolio.trades.record(_trade("2026-08-25", symbol="A", pnl=-100.0, exit_reason="stop hit"))
    portfolio.trades.record(_trade("2026-08-26", symbol="B", pnl=-200.0, exit_reason="1.7% 손절 (-1.7%)"))
    portfolio.trades.record(_trade("2026-08-27", symbol="C", pnl=300.0, exit_reason="확정 익절"))
    risk = RiskManager(config, portfolio)

    text = build_weekly_report(config, portfolio, risk, "PAPER", week_end=week_end)
    assert "손절" in text
    assert "2건" in text  # the two stop-outs grouped together
    assert "확정 익절" in text
