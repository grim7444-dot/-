"""bounce_screener: the opposite of the main screener -- finds today's sharp
crashes instead of uptrends, and feeds them to BounceReversal (2026-09-17,
user request: "config.yaml에 직접 종목 추가. 2번도 진행해줘", i.e. both a
manual config.yaml path AND an automatic screener path for bounce candidates).
"""

from __future__ import annotations

import pandas as pd
import pytest

import screener as screener_module
from screener import DailyScreener, _BOUNCE_CFG_TEMPLATE


def _snapshot(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """A pykrx-shaped snapshot: index is ticker, Korean column names."""
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "종목코드"
    return df


def test_disabled_by_default_returns_nothing():
    config = {"universe": {}}
    screener = DailyScreener(config)
    assert screener.scan_bounce_candidates() == []


def test_finds_a_crashed_liquid_stock_via_the_snapshot_change_column(monkeypatch):
    snap = _snapshot({
        # -9% -- clears the -7% crash_pct bar, liquid enough, priced in range.
        "111111": {
            "시가": 10_000, "고가": 10_200, "저가": 9_000, "종가": 9_100,
            "거래량": 500_000, "거래대금": 5_000_000_000, "등락률": -9.0,
        },
        # -3% -- nowhere near the crash bar, must be excluded.
        "222222": {
            "시가": 10_000, "고가": 10_100, "저가": 9_700, "종가": 9_700,
            "거래량": 500_000, "거래대금": 5_000_000_000, "등락률": -3.0,
        },
    })
    monkeypatch.setattr(screener_module, "_get_market_snapshot", lambda *a, **k: snap)
    monkeypatch.setattr(screener_module, "_ticker_name", lambda t: f"name-{t}")

    config = {
        "universe": {},
        "bounce_screener": {
            "enabled": True, "n_stocks": 5, "crash_pct": 0.07,
            "min_price": 3000, "max_price": 40000,
            "min_trading_value_m": 300, "markets": ["KOSDAQ"],
        },
    }
    results = DailyScreener(config).scan_bounce_candidates()

    assert [t for t, _ in results] == ["111111"]
    ticker, cfg = results[0]
    assert cfg["strategy"] == "bounce"
    assert cfg["name"] == "name-111111"
    assert cfg["market"] == "KOSDAQ"
    assert cfg["_screener"] is True
    assert cfg["_bounce"] is True
    # Params are the strategy's own tuned defaults, not empty.
    assert cfg["params"]["crash_pct"] == _BOUNCE_CFG_TEMPLATE["params"]["crash_pct"]
    # Mutating the returned params dict must not corrupt the shared template.
    cfg["params"]["crash_pct"] = 0.5
    assert _BOUNCE_CFG_TEMPLATE["params"]["crash_pct"] == 0.07


def test_excludes_stocks_already_in_the_universe(monkeypatch):
    snap = _snapshot({
        "111111": {
            "시가": 10_000, "고가": 10_200, "저가": 9_000, "종가": 9_100,
            "거래량": 500_000, "거래대금": 5_000_000_000, "등락률": -9.0,
        },
    })
    monkeypatch.setattr(screener_module, "_get_market_snapshot", lambda *a, **k: snap)
    monkeypatch.setattr(screener_module, "_ticker_name", lambda t: t)

    config = {
        "universe": {"111111": {"name": "already tracked"}},
        "bounce_screener": {"enabled": True, "markets": ["KOSDAQ"]},
    }
    assert DailyScreener(config).scan_bounce_candidates() == []


def test_price_and_trading_value_filters_apply(monkeypatch):
    snap = _snapshot({
        # Crashed enough, but too cheap (wide spread risk).
        "111111": {
            "시가": 2_500, "고가": 2_600, "저가": 2_200, "종가": 2_300,
            "거래량": 500_000, "거래대금": 5_000_000_000, "등락률": -8.0,
        },
        # Crashed enough, priced fine, but too illiquid.
        "222222": {
            "시가": 10_000, "고가": 10_200, "저가": 9_000, "종가": 9_100,
            "거래량": 500, "거래대금": 5_000_000, "등락률": -8.0,
        },
    })
    monkeypatch.setattr(screener_module, "_get_market_snapshot", lambda *a, **k: snap)
    monkeypatch.setattr(screener_module, "_ticker_name", lambda t: t)

    config = {
        "universe": {},
        "bounce_screener": {
            "enabled": True, "min_price": 3000, "max_price": 40000,
            "min_trading_value_m": 300, "markets": ["KOSDAQ"],
        },
    }
    assert DailyScreener(config).scan_bounce_candidates() == []


def test_sorts_biggest_crash_first_and_caps_at_n_stocks(monkeypatch):
    snap = _snapshot({
        "111111": {
            "시가": 10_000, "고가": 10_200, "저가": 9_000, "종가": 9_100,
            "거래량": 500_000, "거래대금": 5_000_000_000, "등락률": -7.5,
        },
        "222222": {
            "시가": 10_000, "고가": 10_200, "저가": 8_500, "종가": 8_800,
            "거래량": 500_000, "거래대금": 5_000_000_000, "등락률": -12.0,
        },
        "333333": {
            "시가": 10_000, "고가": 10_200, "저가": 8_900, "종가": 9_000,
            "거래량": 500_000, "거래대금": 5_000_000_000, "등락률": -10.0,
        },
    })
    monkeypatch.setattr(screener_module, "_get_market_snapshot", lambda *a, **k: snap)
    monkeypatch.setattr(screener_module, "_ticker_name", lambda t: t)

    config = {
        "universe": {},
        "bounce_screener": {"enabled": True, "n_stocks": 2, "markets": ["KOSDAQ"]},
    }
    results = DailyScreener(config).scan_bounce_candidates()
    assert [t for t, _ in results] == ["222222", "333333"]


def test_falls_back_to_per_ticker_history_when_change_column_is_missing(monkeypatch):
    """A pykrx schema change that drops 등락률 must degrade, not silently
    find zero candidates forever."""
    snap = pd.DataFrame(
        {
            "시가": [10_000], "고가": [10_200], "저가": [9_000], "종가": [9_100],
            "거래량": [500_000], "거래대금": [5_000_000_000],
        },
        index=pd.Index(["111111"], name="종목코드"),
    )
    monkeypatch.setattr(screener_module, "_get_market_snapshot", lambda *a, **k: snap)
    monkeypatch.setattr(screener_module, "_ticker_name", lambda t: t)

    hist = pd.DataFrame({"close": [10_000.0, 9_100.0]})  # -9%

    def _fake_history(ticker, fromdate, todate):
        assert ticker == "111111"
        return hist

    monkeypatch.setattr(screener_module, "_fetch_history", _fake_history)

    config = {
        "universe": {},
        "bounce_screener": {"enabled": True, "crash_pct": 0.07, "markets": ["KOSDAQ"]},
    }
    results = DailyScreener(config).scan_bounce_candidates()
    assert [t for t, _ in results] == ["111111"]


def test_falls_back_but_rejects_a_mild_dip(monkeypatch):
    snap = pd.DataFrame(
        {
            "시가": [10_000], "고가": [10_200], "저가": [9_600], "종가": [9_700],
            "거래량": [500_000], "거래대금": [5_000_000_000],
        },
        index=pd.Index(["111111"], name="종목코드"),
    )
    monkeypatch.setattr(screener_module, "_get_market_snapshot", lambda *a, **k: snap)
    monkeypatch.setattr(screener_module, "_ticker_name", lambda t: t)

    hist = pd.DataFrame({"close": [10_000.0, 9_700.0]})  # only -3%
    monkeypatch.setattr(screener_module, "_fetch_history", lambda *a, **k: hist)

    config = {
        "universe": {},
        "bounce_screener": {"enabled": True, "crash_pct": 0.07, "markets": ["KOSDAQ"]},
    }
    assert DailyScreener(config).scan_bounce_candidates() == []


def test_missing_snapshot_is_skipped_not_fatal(monkeypatch):
    monkeypatch.setattr(screener_module, "_get_market_snapshot", lambda *a, **k: None)
    config = {"universe": {}, "bounce_screener": {"enabled": True, "markets": ["KOSDAQ"]}}
    assert DailyScreener(config).scan_bounce_candidates() == []


def test_cover_open_positions_recovers_a_bounce_position_with_bounce_params(tmp_path, monkeypatch):
    """A restart that loses a bounce position's universe entry must not
    silently hand it ORB or pullback_bounce params (wrong stop/target
    fields entirely) -- see the matching _PULLBACK_CFG_TEMPLATE fix this
    mirrors in main._cover_open_positions."""
    import main
    from portfolio import LONG, Portfolio, Position

    monkeypatch.setattr(screener_module, "_ticker_name", lambda t: "테스트종목")

    portfolio = Portfolio(
        state_path=tmp_path / "state.json",
        trades_path=tmp_path / "trades.csv",
        daily_path=tmp_path / "daily_pnl.csv",
        mode_label="DRY-RUN",
    )
    portfolio.open_position(Position(
        symbol="999999", side=LONG, qty=10, entry_price=9_100.0,
        stop_price=8_918.0, stop_distance=182.0, strategy="bounce",
    ))

    config = {"universe": {}}
    updated = main._cover_open_positions(config, portfolio)

    entry = updated["universe"]["999999"]
    assert entry["strategy"] == "bounce"
    assert "crash_pct" in entry["params"]
    assert "swing_lookback" not in entry["params"]
    assert "range_minutes" not in entry["params"]
