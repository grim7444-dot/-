"""Shared fixtures.

Two things matter for every test in this suite:

* the process environment is scrubbed of ``ALPACA_*`` so a developer's real
  ``.env`` can never turn a test into a live-trading attempt;
* nothing touches the network — bars come from the deterministic synthetic
  generator and the broker is always :class:`broker.DryRunBroker`.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV_KEYS = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_PAPER",
    "ALPACA_LIVE_CONFIRM",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove every credential/mode variable before each test."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return {}


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """An isolated working directory holding a copy of config.yaml."""
    (tmp_path / "config.yaml").write_text(
        (ROOT / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def config():
    from settings import load_config

    return load_config(ROOT / "config.yaml")


def make_bars(
    n: int = 400,
    start_price: float = 100.0,
    seed: int = 7,
    freq: str = "15min",
    volatility: float = 0.004,
) -> pd.DataFrame:
    """Deterministic OHLCV bars with mean-reverting swings and volume spikes."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(
        start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc), periods=n, freq=freq
    )
    phase = np.linspace(0.0, 8.0 * np.pi, n)
    log_returns = 0.9 * volatility * np.sin(phase) + rng.normal(0.0, volatility, n)
    close = start_price * np.exp(np.cumsum(log_returns))

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, volatility, n))
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    volume = rng.lognormal(mean=np.log(1_000_000), sigma=0.4, size=n)

    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


@pytest.fixture
def bars():
    return make_bars()


@pytest.fixture
def portfolio(tmp_path):
    from portfolio import Portfolio

    return Portfolio(
        state_path=tmp_path / "state.json",
        trades_path=tmp_path / "trades.csv",
        daily_path=tmp_path / "daily_pnl.csv",
        mode_label="PAPER",
    )


class RecordingBroker:
    """Minimal broker double that records teardown calls."""

    def __init__(self) -> None:
        self.cancelled = 0
        self.closed = 0
        self.orders: list[tuple] = []

    def cancel_all_orders(self) -> int:
        self.cancelled += 1
        return 0

    def close_all_positions(self) -> int:
        self.closed += 1
        return 0

    def submit_order(self, **kwargs):
        self.orders.append(kwargs)
        raise AssertionError("tests must never submit an order")


@pytest.fixture
def recording_broker():
    return RecordingBroker()
