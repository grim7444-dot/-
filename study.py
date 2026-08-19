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

#: A stock contributing at or below this many events tells you about its own
#: luck rather than about the pattern, so the report also shows the pooled
#: result without those stocks.
MIN_EVENTS_PER_STOCK = 3


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
        for label, pooled in (
            ("to next open ", pooled_open),
            ("to next close", pooled_close),
        ):
            lines.append(
                f"    {label}: mean {pooled['mean']:+.2%}   "
                f"median {pooled['median']:+.2%}   "
                f"win {pooled['win_rate']:.0%}   "
                f"t {pooled['t_stat']:+.2f}"
            )
        lines.append("")
        lines.append(
            "    t is the mean divided by its own standard error. Below about 2,"
        )
        lines.append(
            "    a positive average is within what chance produces at this sample size."
        )
        # Concentration check. A pooled average carried by a stock with one or
        # two occurrences is that stock's luck, not the pattern's behaviour.
        thin = [s for s in stats if 0 < s.count <= MIN_EVENTS_PER_STOCK]
        if thin and any(s.count > MIN_EVENTS_PER_STOCK for s in stats):
            deep = [s for s in stats if s.count > MIN_EVENTS_PER_STOCK]
            robust_open = _pool(deep, cost_pct, to_open=True)
            robust_close = _pool(deep, cost_pct, to_open=False)
            dropped = sum(s.count for s in thin)
            lines.append("")
            lines.append(
                f"    Dropping the {len(thin)} stock(s) with "
                f"{MIN_EVENTS_PER_STOCK} or fewer occurrences "
                f"({dropped} of {total_events} events):"
            )
            lines.append(
                f"      to next open : mean {robust_open['mean']:+.2%}   "
                f"win {robust_open['win_rate']:.0%}   t {robust_open['t_stat']:+.2f}"
            )
            lines.append(
                f"      to next close: mean {robust_close['mean']:+.2%}   "
                f"win {robust_close['win_rate']:.0%}   t {robust_close['t_stat']:+.2f}"
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
        return {"mean": 0.0, "median": 0.0, "win_rate": 0.0, "t_stat": 0.0, "n": 0}
    n = len(values)
    mean = sum(values) / n
    ordered = sorted(values)
    middle = n // 2
    median = ordered[middle] if n % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    # Standard error of the mean. Without it "the average is positive" and
    # "the average is reliably positive" are impossible to tell apart, which
    # is the whole failure mode this study exists to avoid.
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std_error = (variance / n) ** 0.5
    else:
        std_error = float("inf")
    return {
        "mean": mean,
        "median": median,
        "win_rate": sum(1 for v in values if v > 0) / n,
        "std_error": std_error,
        "t_stat": mean / std_error if std_error else 0.0,
        "n": float(n),
    }


def _verdict(
    to_open: Mapping[str, float], to_close: Mapping[str, float], count: int
) -> list[str]:
    """Say what the numbers support, including when that is 'not much'.

    A positive mean is the weakest of the available signals and the easiest to
    be fooled by, so it is the last thing considered rather than the first.
    Whether the typical trade wins, how often any trade wins, and whether the
    average survives its own error bar all come first.
    """
    lines: list[str] = []
    best_label, best = max(
        (("selling at the next open", to_open), ("selling at the next close", to_close)),
        key=lambda pair: pair[1]["mean"],
    )

    if best["mean"] <= 0:
        lines.append(
            "After costs the pattern loses money on both exits. Whatever the bounces"
        )
        lines.append("looked like, the ones that did not bounce cost more.")
        return lines

    if best.get("median", 0.0) <= 0:
        lines.append(
            f"The average is positive but the median is {best['median']:+.2%}: more "
            "than half"
        )
        lines.append(
            "of these trades lost money, and the average is held up by a few large"
        )
        lines.append(
            "winners. That is a lottery ticket, not an edge -- and it is exactly the"
        )
        lines.append(
            "shape that memory reports as 'this happens a lot', because the outliers"
        )
        lines.append("are the ones worth remembering.")
        lines.append("")

    if best.get("win_rate", 0.0) < 0.5:
        lines.append(
            f"It wins {best['win_rate']:.0%} of the time. Fewer than half the trades"
        )
        lines.append(
            "are profitable, so position sizing has to survive long losing runs."
        )
        lines.append("")

    t_stat = abs(best.get("t_stat", 0.0))
    if t_stat < 2.0:
        lines.append(
            f"t = {best.get('t_stat', 0.0):+.2f} on {count} occurrences: the average is"
        )
        lines.append(
            "inside the range chance produces. This history cannot tell a real edge"
        )
        lines.append("from a lucky one; more data would be needed, not more confidence.")
    else:
        lines.append(
            f"t = {best.get('t_stat', 0.0):+.2f}: the average survives its own error bar."
        )
        lines.append(f"The result is carried by {best_label}.")

    lines.append("")
    lines.append(
        "Every parameter set tried raises the bar. Searching down_days and drop"
    )
    lines.append(
        "until something looks good will find something eventually, on data with"
    )
    lines.append(
        "no edge in it at all -- so a t of 2 after one attempt is not the same"
    )
    lines.append(
        "result as a t of 2 after five. Count the attempts honestly, and prefer"
    )
    lines.append("a setting chosen for a reason over the one that scored best.")

    if to_close["mean"] > to_open["mean"] * 2 and to_open["mean"] > 0:
        lines.append("")
        lines.append(
            "Most of the move happens during the next session, not in the overnight"
        )
        lines.append(
            "gap. That makes this a swing trade held through a full day -- with a"
        )
        lines.append(
            "day's worth of exposure -- rather than the close-to-open trade it was"
        )
        lines.append("proposed as.")
    return lines


def parse_codes(raw: str) -> list[str]:
    """Split a pasted list of KRX codes on whatever separated them.

    People paste these with dots, commas, spaces or newlines between them, and
    sometimes with a leading zero dropped by a spreadsheet. Codes are six
    digits, so a shorter run of digits is padded rather than rejected.
    """
    import re

    out: list[str] = []
    for chunk in re.split(r"[^0-9]+", raw or ""):
        if not chunk:
            continue
        if len(chunk) > 6:
            # A run this long is several codes with nothing between them.
            for i in range(0, len(chunk), 6):
                piece = chunk[i : i + 6]
                if piece:
                    out.append(piece.zfill(6))
            continue
        out.append(chunk.zfill(6))
    seen: set[str] = set()
    unique: list[str] = []
    for code in out:
        if code not in seen:
            seen.add(code)
            unique.append(code)
    return unique


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
