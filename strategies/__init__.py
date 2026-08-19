"""Strategy registry.

``build_strategies(config)`` turns the ``universe:`` block of ``config.yaml``
into ready-to-run strategy objects, so adding a ninth stock is a config edit
rather than a code edit.

Every strategy is built with ``allow_short`` tied to ``risk.long_only``. Retail
short selling of KRX equities is not practically available, so the default is
long-only: a short signal closes a position but never opens one.
"""

from __future__ import annotations

from typing import Any, Mapping

from strategies.base import Action, Signal, Strategy
from strategies.breakout import Breakout
from strategies.close_auction import CloseAuction
from strategies.mean_reversion import MeanReversion
from strategies.scalping import Scalping
from strategies.trend_following import TrendFollowing

STRATEGY_CLASSES: dict[str, type[Strategy]] = {
    "mean_reversion": MeanReversion,
    "breakout": Breakout,
    "trend_following": TrendFollowing,
    "scalping": Scalping,
    "close_auction": CloseAuction,
}

__all__ = [
    "Action",
    "Signal",
    "Strategy",
    "Breakout",
    "CloseAuction",
    "MeanReversion",
    "Scalping",
    "TrendFollowing",
    "STRATEGY_CLASSES",
    "build_strategy",
    "build_strategies",
]


def build_strategy(
    code: str,
    asset_cfg: Mapping[str, Any],
    risk_cfg: Mapping[str, Any],
) -> Strategy:
    name = asset_cfg.get("strategy")
    try:
        cls = STRATEGY_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r} for {code}; expected one of {sorted(STRATEGY_CLASSES)}"
        ) from None
    params = dict(asset_cfg.get("params") or {})
    params.setdefault("allow_short", not bool(risk_cfg.get("long_only", True)))
    return cls(
        symbol=code,
        timeframe=asset_cfg.get("timeframe", "1Day"),
        atr_period=int(risk_cfg.get("atr_period", 14)),
        hard_stop_atr_mult=float(risk_cfg.get("hard_stop_atr_mult", 1.0)),
        **params,
    )


def build_strategies(config: Mapping[str, Any]) -> dict[str, Strategy]:
    risk_cfg = config.get("risk") or {}
    out: dict[str, Strategy] = {}
    for code, asset_cfg in (config.get("universe") or {}).items():
        if not asset_cfg.get("enabled", True):
            continue
        out[str(code)] = build_strategy(str(code), asset_cfg, risk_cfg)
    return out
