"""Does the bounce actually happen?

A common belief about KRX small caps is that a stock which falls hard for two
sessions in a row tends to snap back on the third. It is a plausible thing to
believe and an easy thing to misremember: the snap-backs are memorable and the
names that simply kept falling are not, so recollection systematically
overstates the edge. That is what this module exists to settle -- on the user's
own stocks, over their own history, rather than in the abstract.

The measurement is deliberately literal. Find every run of ``down_days``
consecutive losing sessions whose combined fall reaches ``drop_pct``, then
record what the next session did: buy at the close of the last down day, sell
at the next open and at the next close. Both exits are reported because they
answer different questions -- the open is what an overnight trade gets, the
close is what a full extra day gets, and if only the second one pays then the
pattern is not the overnight bounce it is remembered as.

Costs are subtracted, not mentioned in passing. A round trip on KRX is about
0.38%, and a bounce that averages +0.3% is a losing trade dressed up as a
winning observation.

**Buying an actual limit-down close is usually not possible.** At the limit
there is no bid to hit: the book is sell orders all the way down, which is what
put the stock there. ``--limit-down`` measures the pattern anyway, because
knowing whether the bounce is real matters even when this particular entry is
not reachable, but the report says so rather than implying a tradeable result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

logger = logging.getLogger("bot.study")

#: A daily fall this size can only be the -30% limit, allowing for the tick
#: grid and for a close that comes off the limit by a tick or two.
LIMIT_DOWN_RETURN = -0.28


@dataclass
class BounceEvent:
    """One occurrence of the pattern, and what happened next."""

    code: str
    trigger_date: pd.Timestamp
    #: Combined return over the qualifying down days.
    drop_pct: float
    entry_close: float
    next_open: float
    next_close: float

    @property
    def to_open_pct(self) -> float:
        return self.next_open / self.entry_close - 1.0

    @property
    def to_close_pct(self) -> float:
        return self.next_close / self.entry_close - 1.0


@dataclass
class BounceStats:
    code: str
    name: str = ""
    sessions: int = 0
    events: list[BounceEvent] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.events)

    def _summarise(self, values: Sequence[float], cost_pct: float) -> dict[str, float]:
        if not values:
            return {}
        net = [v - cost_pct for v in values]
        ordered = sorted(net)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return {
            "mean": sum(net) / len(net),
            "median": median,
            "win_rate": sum(1 for v in net if v > 0) / len(net),
            "best": max(net),
            "worst": min(net),
            "total": sum(net),
        }

    def to_open(self, cost_pct: float) -> dict[str, float]:
        return self._summarise([e.to_open_pct for e in self.events], cost_pct)

    def to_close(self, cost_pct: float) -> dict[str, float]:
        return self._summarise([e.to_close_pct for e in self.events], cost_pct)


def find_bounce_events(
    code: str,
    bars: pd.DataFrame,
    down_days: int = 2,
    drop_pct: float = 0.15,
    limit_down: bool = False,
) -> list[BounceEvent]:
    """Every run of *down_days* losing sessions that fell at least *drop_pct*.

    The trigger bar is the last down day. The next bar must exist -- a pattern
    whose outcome has not happened yet is not evidence of anything.
    """
    if len(bars) < down_days + 2:
        return []
    closes = bars["close"].astype(float)
    returns = closes.pct_change()

    events: list[BounceEvent] = []
    for i in range(down_days, len(bars) - 1):
        window = returns.iloc[i - down_days + 1 : i + 1]
        if window.isna().any():
            continue
        if limit_down:
            if not (window <= LIMIT_DOWN_RETURN).all():
                continue
        elif not (window < 0).all():
            continue

        combined = float(closes.iloc[i] / closes.iloc[i - down_days] - 1.0)
        if combined > -abs(drop_pct):
            continue

        nxt = bars.iloc[i + 1]
        entry = float(closes.iloc[i])
        if entry <= 0 or float(nxt["open"]) <= 0:
            continue
        events.append(
            BounceEvent(
                code=code,
                trigger_date=bars.index[i],
                drop_pct=combined,
                entry_close=entry,
                next_open=float(nxt["open"]),
                next_close=float(nxt["close"]),
            )
        )
    return events


def format_bounce_study(
    stats: Sequence[BounceStats],
    down_days: int,
    drop_pct: float,
    cost_pct: float,
    limit_down: bool,
) -> str:
    width = 100
    lines = ["=" * width]
    title = "TWO-DAY DROP, NEXT-DAY BOUNCE" if down_days == 2 else "DROP AND BOUNCE"
    lines += [title.center(width), "=" * width, ""]

    if limit_down:
        lines.append(
            f"  Pattern: {down_days} consecutive sessions closing at the -30% limit."
        )
        lines.append("")
        lines.append("  NOTE: buying a limit-down close is usually not possible. At the")
        lines.append("  limit the book is sell orders with no bid to hit -- that is what")
        lines.append("  put the stock there. These numbers say whether the bounce is real,")
        lines.append("  not whether this entry can be filled.")
    else:
        lines.append(
            f"  Pattern: {down_days} consecutive down sessions falling "
            f"{drop_pct:.0%} or more in total."
        )
    lines.append("")
    lines.append(
        f"  Entry at the trigger day's close, net of {cost_pct:.2%} round-trip costs."
    )
    lines.append("")

    header = (
        f"  {'code':<8} {'name':<14} {'sessions':>9} {'events':>7}"
        f"{'  |':>3} {'mean':>8} {'median':>8} {'win':>6} {'best':>8} {'worst':>8}"
    )
    lines.append("  -- Sell at the NEXT OPEN " + "-" * (width - 29))
    lines.append(header)
    lines.append("  " + "-" * (width - 4))
    for s in stats:
        lines.append(_row(s, s.to_open(cost_pct)))

    lines.append("")
    lines.append("  -- Sell at the NEXT CLOSE " + "-" * (width - 30))
    lines.append(header)
    lines.append("  " + "-" * (width - 4))
    for s in stats:
        lines.append(_row(s, s.to_close(cost_pct)))

    total_events = sum(s.count for s in stats)
    lines += ["", "  " + "-" * (width - 4)]
    if not total_events:
        lines.append(
            "  The pattern never occurred in this history. Nothing here says it does"
        )
        lines.append(
            "  not work -- only that these stocks over this window cannot answer it."
        )
    else:
        pooled_open = _pool(stats, cost_pct, to_open=True)
        pooled_close = _pool(stats, cost_pct, to_open=False)
        lines.append(f"  {total_events} occurrences across {len(stats)} stocks.")
        lines.append(
            f"    to next open : mean {pooled_open['mean']:+.2%}  "
            f"win {pooled_open['win_rate']:.0%}"
        )
        lines.append(
            f"    to next close: mean {pooled_close['mean']:+.2%}  "
            f"win {pooled_close['win_rate']:.0%}"
        )
        lines.append("")
        verdict = _verdict(pooled_open, pooled_close, total_events)
        for line in verdict:
            lines.append(f"  {line}")
    lines.append("=" * width)
    return "\n".join(lines)


def _row(stats: BounceStats, summary: Mapping[str, float]) -> str:
    name = (stats.name or "")[:13]
    if not summary:
        return (
            f"  {stats.code:<8} {name:<14} {stats.sessions:>9d} {stats.count:>7d}"
            f"{'  |':>3} {'-':>8} {'-':>8} {'-':>6} {'-':>8} {'-':>8}"
        )
    return (
        f"  {stats.code:<8} {name:<14} {stats.sessions:>9d} {stats.count:>7d}"
        f"{'  |':>3} {summary['mean']:>+8.2%} {summary['median']:>+8.2%} "
        f"{summary['win_rate']:>6.0%} {summary['best']:>+8.2%} {summary['worst']:>+8.2%}"
    )


def _pool(
    stats: Sequence[BounceStats], cost_pct: float, to_open: bool
) -> dict[str, float]:
    values = [
        (e.to_open_pct if to_open else e.to_close_pct) - cost_pct
        for s in stats
        for e in s.events
    ]
    if not values:
        return {"mean": 0.0, "win_rate": 0.0}
    return {
        "mean": sum(values) / len(values),
        "win_rate": sum(1 for v in values if v > 0) / len(values),
    }


def _verdict(
    to_open: Mapping[str, float], to_close: Mapping[str, float], count: int
) -> list[str]:
    """Say what the numbers support, including when that is 'not much'."""
    lines: list[str] = []
    if count < 20:
        lines.append(
            f"Only {count} occurrences. That is too few to separate an edge from "
            "luck;"
        )
        lines.append(
            "a run of good ones proves nothing at this sample size. Treat the "
            "numbers"
        )
        lines.append("above as a description of the past, not as an estimate.")
        lines.append("")
    best = max(to_open["mean"], to_close["mean"])
    if best <= 0:
        lines.append(
            "After costs the pattern loses money on both exits. Whatever the "
            "bounces"
        )
        lines.append("looked like, the ones that did not bounce cost more.")
    elif to_open["mean"] > 0 and to_close["mean"] <= 0:
        lines.append(
            "The gain is in the overnight gap and is given back during the next"
        )
        lines.append("session. That is an overnight trade, not a swing trade.")
    else:
        lines.append(
            "Positive after costs on this history. Worth testing further before "
            "it"
        )
        lines.append(
            "trades real money -- a positive average over few events is still "
            "mostly noise."
        )
    return lines


def run_bounce_study(
    universe: Mapping[str, Mapping[str, Any]],
    market_data,
    months: int = 24,
    down_days: int = 2,
    drop_pct: float = 0.15,
    limit_down: bool = False,
) -> list[BounceStats]:
    from datetime import datetime

    from data import months_to_start
    from market.calendar import KST
    from market.rules import KOSPI

    end = datetime.now(KST)
    start = months_to_start(months, end)

    out: list[BounceStats] = []
    for code, cfg in universe.items():
        code = str(code)
        if not cfg.get("enabled", True):
            continue
        barset = market_data.get_bars(
            code=code,
            timeframe="1Day",
            start=start,
            end=end,
            market=str(cfg.get("market") or KOSPI),
        )
        stats = BounceStats(
            code=code, name=str(cfg.get("name") or ""), sessions=len(barset.bars)
        )
        if barset.synthetic:
            logger.warning("%s: synthetic bars - excluded from the study", code)
        else:
            stats.events = find_bounce_events(
                code, barset.bars, down_days, drop_pct, limit_down
            )
        out.append(stats)
    return out
