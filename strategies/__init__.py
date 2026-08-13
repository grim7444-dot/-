"""Strategy registry.

``build_strategies(config)`` turns the ``assets:`` block of ``config.yaml``
into ready-to-run strategy objects, so adding a sixth asset is a config edit
rather than a code edit.
"""

from __future__ import annotations

from typing import Any, Mapping

from strategies.base import Action, Signal, Strategy
from strategies.breakout import Breakout
from strategies.mean_reversion import MeanReversion
from strategies.trend_following import TrendFollowing

STRATEGY_CLASSES: dict[str, type[Strategy]] = {
    "mean_reversion": MeanReversion,
    "breakout": Breakout,
    "trend_following": TrendFollowing,
}

__all__ = [
    "Action",
    "Signal",
    "Strategy",
    "Breakout",
    "MeanReversion",
    "TrendFollowing",
    "STRATEGY_CLASSES",
    "build_strategy",
    "build_strategies",
]


def build_strategy(symbol: str, asset_cfg: Mapping[str, Any], risk_cfg: Mapping[str, Any]) -> Strategy:
    name = asset_cfg.get("strategy")
    try:
        cls = STRATEGY_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r} for {symbol}; "
            f"expected one of {sorted(STRATEGY_CLASSES)}"
        ) from None
    params = dict(asset_cfg.get("params") or {})
    return cls(
        symbol=symbol,
        timeframe=asset_cfg.get("timeframe", "1Hour"),
        atr_period=int(risk_cfg.get("atr_period", 14)),
        hard_stop_atr_mult=float(risk_cfg.get("hard_stop_atr_mult", 1.0)),
        **params,
    )


def build_strategies(config: Mapping[str, Any]) -> dict[str, Strategy]:
    risk_cfg = config.get("risk") or {}
    out: dict[str, Strategy] = {}
    for symbol, asset_cfg in (config.get("assets") or {}).items():
        if not asset_cfg.get("enabled", True):
            continue
        out[symbol] = build_strategy(symbol, asset_cfg, risk_cfg)
    return out
