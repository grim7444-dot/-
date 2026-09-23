"""MarketData.get_index_change_pct -- KOSPI/KOSDAQ day-over-day change used
to soften the profit-lock target on a down day (2026-09-10, user request:
"장이 안좋을때는 1%수익이라도 낼줄 알아야지"). None means "unknown" and must
never be read by a caller as "market is down" -- a pykrx hiccup should leave
the normal, higher profit target in place.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from data import MarketData
from market.calendar import KST, KrxCalendar
from market.rules import KOSDAQ, KOSPI


class _FakeIndexStock:
    def __init__(self, frame):
        self._frame = frame

    def get_index_ohlcv(self, fromdate, todate, ticker):
        return self._frame


def _index_frame(prev_close: float, last_close: float) -> pd.DataFrame:
    """Two daily rows dated up through *today* -- is_current() only demands
    the last row reach the most recent business day, so real "today" always
    satisfies it regardless of whether the market happens to be open."""
    today = datetime.now(KST)
    yesterday = today - timedelta(days=1)
    return pd.DataFrame(
        {
            "시가": [prev_close, last_close],
            "고가": [prev_close, last_close],
            "저가": [prev_close, last_close],
            "종가": [prev_close, last_close],
            "거래량": [1_000_000, 1_000_000],
        },
        index=pd.to_datetime([yesterday.date(), today.date()]),
    )


def _market_data(tmp_path, monkeypatch, frame=None, raise_error=False, **kwargs):
    import data as data_module

    if raise_error:
        monkeypatch.setattr(
            data_module, "_import_pykrx_stock",
            lambda: (_ for _ in ()).throw(RuntimeError("KRX down")),
        )
    else:
        monkeypatch.setattr(data_module, "_import_pykrx_stock", lambda: _FakeIndexStock(frame))
    return MarketData(calendar=KrxCalendar(), cache_dir=tmp_path / "cache", **kwargs)


def test_a_down_index_reports_a_negative_change(tmp_path, monkeypatch):
    md = _market_data(tmp_path, monkeypatch, frame=_index_frame(2_600.0, 2_574.0))
    change = md.get_index_change_pct(KOSPI, use_cache=False)
    assert change == pytest.approx((2_574.0 - 2_600.0) / 2_600.0)
    assert change < 0


def test_an_up_index_reports_a_positive_change(tmp_path, monkeypatch):
    md = _market_data(tmp_path, monkeypatch, frame=_index_frame(2_600.0, 2_626.0))
    change = md.get_index_change_pct(KOSPI, use_cache=False)
    assert change > 0


def test_pykrx_unavailable_reports_unknown_not_down(tmp_path, monkeypatch):
    md = _market_data(tmp_path, monkeypatch, raise_error=True)
    assert md.get_index_change_pct(KOSPI, use_cache=False) is None


def test_an_unmapped_market_label_reports_unknown(tmp_path, monkeypatch):
    md = _market_data(tmp_path, monkeypatch, frame=_index_frame(2_600.0, 2_574.0))
    assert md.get_index_change_pct("SOMETHING_ELSE", use_cache=False) is None


def test_network_disabled_reports_unknown_without_calling_pykrx(tmp_path, monkeypatch):
    import data as data_module

    def boom():
        raise AssertionError("pykrx must not be touched when allow_network is False")

    monkeypatch.setattr(data_module, "_import_pykrx_stock", boom)
    md = MarketData(calendar=KrxCalendar(), cache_dir=tmp_path / "cache", allow_network=False)
    assert md.get_index_change_pct(KOSPI, use_cache=False) is None


def test_a_cached_change_is_reused_without_hitting_pykrx_again(tmp_path, monkeypatch):
    md = _market_data(tmp_path, monkeypatch, frame=_index_frame(2_600.0, 2_574.0))
    first = md.get_index_change_pct(KOSPI)
    assert first is not None

    import data as data_module

    def boom(*a, **k):
        raise AssertionError("a cache hit must not call pykrx again")

    monkeypatch.setattr(data_module, "_import_pykrx_stock", boom)
    second = md.get_index_change_pct(KOSPI)
    assert second == pytest.approx(first)


def test_kosdaq_and_kospi_are_tracked_independently(tmp_path, monkeypatch):
    """A KOSDAQ stock must never read the KOSPI index's change, or vice versa."""
    import data as data_module

    frames = {KOSPI: _index_frame(2_600.0, 2_574.0), KOSDAQ: _index_frame(800.0, 808.0)}

    class _Router:
        def get_index_ohlcv(self, fromdate, todate, ticker):
            from data import INDEX_TICKERS

            for market, t in INDEX_TICKERS.items():
                if t == ticker:
                    return frames[market]
            raise AssertionError(f"unexpected ticker {ticker}")

    monkeypatch.setattr(data_module, "_import_pykrx_stock", lambda: _Router())
    md = MarketData(calendar=KrxCalendar(), cache_dir=tmp_path / "cache")

    assert md.get_index_change_pct(KOSPI, use_cache=False) < 0
    assert md.get_index_change_pct(KOSDAQ, use_cache=False) > 0
