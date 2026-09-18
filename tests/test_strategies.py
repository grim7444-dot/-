import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_bot.strategies import MACrossoverStrategy, RSIStrategy, MACDStrategy, Signal


def make_df(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="1h")
    return pd.DataFrame({
        "open": prices, "high": prices, "low": prices,
        "close": prices, "volume": [1000.0] * len(prices),
    }, index=idx)


def test_ma_crossover_golden():
    # 초반엔 하락 → 후반엔 상승 (골든 크로스 유도)
    prices = list(range(50, 0, -1)) + list(range(0, 80))
    df = make_df(prices)
    strategy = MACrossoverStrategy({"fast_period": 5, "slow_period": 20})
    signal = strategy.analyze(df)
    assert signal.signal in (Signal.BUY, Signal.HOLD)


def test_rsi_oversold():
    prices = [100] * 10 + [50] * 5
    df = make_df(prices)
    strategy = RSIStrategy({"rsi_period": 5, "rsi_oversold": 40, "rsi_overbought": 70})
    signal = strategy.analyze(df)
    assert signal.signal == Signal.BUY


def test_rsi_overbought():
    prices = [50] * 10 + [100] * 5
    df = make_df(prices)
    strategy = RSIStrategy({"rsi_period": 5, "rsi_oversold": 30, "rsi_overbought": 60})
    signal = strategy.analyze(df)
    assert signal.signal == Signal.SELL


def test_macd_strategy_runs():
    prices = [float(i) + np.sin(i * 0.5) * 10 for i in range(100)]
    df = make_df(prices)
    strategy = MACDStrategy({"macd_fast": 12, "macd_slow": 26, "macd_signal": 9})
    signal = strategy.analyze(df)
    assert signal.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
    assert 0.0 <= signal.confidence <= 1.0
