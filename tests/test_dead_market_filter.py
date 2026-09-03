"""The volatility/volume "dead market" entry filter.

User request (2026-08-27, Korean): 거래량과 변동성이 죽은 횡보장에서는
진입 신호가 나와도 봇이 매매하지 않도록 제한하여 수수료 낭비 방지. A
strategy's own entry conditions can line up technically while the stock
simply isn't moving -- this blocks a fresh entry in that case. It only
ever gates ENTER signals (see the call site in main.py's _process_code,
placed after the entry-window check and before _submit_entry); it never
touches an exit.
"""

from __future__ import annotations

import pandas as pd

from market.calendar import KST


def _bars(n: int, price_range_pct: float, volumes: list[float]) -> pd.DataFrame:
    """n bars, each with a fixed high/low spread around a constant close."""
    idx = pd.date_range("2026-08-27 09:00", periods=n, freq="1min", tz=KST)
    base = 10_000.0
    close = [base] * n
    high = [base * (1 + price_range_pct)] * n
    low = [base * (1 - price_range_pct)] * n
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volumes},
        index=idx,
    )


N = 25  # enough for the default atr_period=14 and volume_lookback=20


def test_blocks_entry_when_volatility_is_dead():
    from main import _dead_market_reason

    bars = _bars(N, price_range_pct=0.0005, volumes=[500_000] * N)  # ATR% ~0.1%
    reason = _dead_market_reason(bars, {})
    assert reason is not None
    assert "변동성" in reason


def test_allows_entry_when_volatility_and_volume_are_both_healthy():
    from main import _dead_market_reason

    bars = _bars(N, price_range_pct=0.01, volumes=[500_000] * N)  # ATR% ~2%
    assert _dead_market_reason(bars, {}) is None


def test_blocks_entry_on_a_sudden_volume_die_off_even_with_healthy_volatility():
    from main import _dead_market_reason

    volumes = [100_000] * (N - 5) + [5_000] * 5  # healthy ATR, but volume just collapsed
    bars = _bars(N, price_range_pct=0.01, volumes=volumes)
    reason = _dead_market_reason(bars, {})
    assert reason is not None
    assert "거래량" in reason


def test_fails_open_with_not_enough_history_yet():
    """Blocking every entry during a strategy's own warm-up would be worse than no filter."""
    from main import _dead_market_reason

    bars = _bars(10, price_range_pct=0.0005, volumes=[500_000] * 10)  # obviously "dead" data
    assert _dead_market_reason(bars, {}) is None  # too few bars to judge -- fail open


def test_disabled_lets_everything_through():
    from main import _dead_market_reason

    bars = _bars(N, price_range_pct=0.0005, volumes=[5_000] * N)  # dead on both counts
    assert _dead_market_reason(bars, {"enabled": False}) is None


def test_thresholds_are_configurable():
    from main import _dead_market_reason

    bars = _bars(N, price_range_pct=0.0005, volumes=[500_000] * N)  # dead under defaults
    # A lenient enough ATR floor must let it through despite the default blocking it.
    lenient_cfg = {"min_atr_pct": 0.0001}
    assert _dead_market_reason(bars, lenient_cfg) is None
