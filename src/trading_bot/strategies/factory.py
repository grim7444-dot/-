from ..config import Config
from .base import BaseStrategy
from .ma_crossover import MACrossoverStrategy
from .rsi import RSIStrategy
from .macd import MACDStrategy

_STRATEGIES = {
    "ma_crossover": MACrossoverStrategy,
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
}


def create_strategy(config: Config) -> BaseStrategy:
    cls = _STRATEGIES.get(config.strategy_name)
    if cls is None:
        raise ValueError(f"알 수 없는 전략: {config.strategy_name}. 사용 가능: {list(_STRATEGIES.keys())}")
    return cls(config.strategy_params)
