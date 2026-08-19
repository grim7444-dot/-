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
from typing import Any, Callable, Mapping, Sequence

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
    #: Every session's close-to-next-open return, by date. The control the
    #: whole study needs: a pattern that buys a stock which rose all year
    #: shows that stock's drift, not the pattern's edge, and only a
    #: comparison against holding it can tell the two apart.
    baseline: "pd.Series | None" = None

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


def find_strength_events(
    code: str,
    bars: pd.DataFrame,
    close_strength: float = 0.7,
    trend_ema: int = 20,
    volume_mult: float = 1.2,
    volume_period: int = 20,
    **_ignored,
) -> list[BounceEvent]:
    """Days that closed strong, in an uptrend, on volume.

    The mirror image of the drop pattern, and the entry the configured
    close_auction strategy actually uses. It gets measured the same way for
    the same reason: a rule nobody has held to the standard is not a better
    rule, only an untested one.
    """
    from indicators import ema, rolling_mean_volume

    warmup = max(trend_ema, volume_period) + 2
    if len(bars) < warmup + 2:
        return []
    closes = bars["close"].astype(float)
    trend = ema(closes, trend_ema)
    avg_volume = rolling_mean_volume(bars, volume_period).shift(1)

    events: list[BounceEvent] = []
    for i in range(warmup, len(bars) - 1):
        high = float(bars["high"].iloc[i])
        low = float(bars["low"].iloc[i])
        close = float(closes.iloc[i])
        span = high - low
        if span <= 0 or close <= 0:
            continue
        if (close - low) / span < close_strength:
            continue
        level = trend.iloc[i]
        if pd.isna(level) or close <= float(level):
            continue
        average = avg_volume.iloc[i]
        if pd.isna(average) or float(average) <= 0:
            continue
        if float(bars["volume"].iloc[i]) < volume_mult * float(average):
            continue

        nxt = bars.iloc[i + 1]
        if float(nxt["open"]) <= 0:
            continue
        events.append(
            BounceEvent(
                code=code,
                trigger_date=bars.index[i],
                drop_pct=(close - low) / span,
                entry_close=close,
                next_open=float(nxt["open"]),
                next_close=float(nxt["close"]),
            )
        )
    return events


def find_volume_spike_events(
    code: str,
    bars: pd.DataFrame,
    volume_mult: float = 5.0,
    volume_period: int = 20,
    **_ignored,
) -> list[BounceEvent]:
    """Days whose volume dwarfed the recent average, whatever the price did."""
    from indicators import rolling_mean_volume

    warmup = volume_period + 2
    if len(bars) < warmup + 2:
        return []
    avg = rolling_mean_volume(bars, volume_period).shift(1)
    closes = bars["close"].astype(float)

    events: list[BounceEvent] = []
    for i in range(warmup, len(bars) - 1):
        average = avg.iloc[i]
        if pd.isna(average) or float(average) <= 0:
            continue
        ratio = float(bars["volume"].iloc[i]) / float(average)
        if ratio < volume_mult:
            continue
        nxt = bars.iloc[i + 1]
        if float(closes.iloc[i]) <= 0 or float(nxt["open"]) <= 0:
            continue
        events.append(
            BounceEvent(
                code=code,
                trigger_date=bars.index[i],
                drop_pct=ratio,
                entry_close=float(closes.iloc[i]),
                next_open=float(nxt["open"]),
                next_close=float(nxt["close"]),
            )
        )
    return events


def find_ma_cross_events(
    code: str,
    bars: pd.DataFrame,
    fast: int = 5,
    slow: int = 20,
    **_ignored,
) -> list[BounceEvent]:
    """The session a fast moving average crossed above a slow one."""
    from indicators import ema

    warmup = slow + 2
    if len(bars) < warmup + 2:
        return []
    closes = bars["close"].astype(float)
    fast_ma = ema(closes, fast)
    slow_ma = ema(closes, slow)

    events: list[BounceEvent] = []
    for i in range(warmup, len(bars) - 1):
        now_f, now_s = fast_ma.iloc[i], slow_ma.iloc[i]
        was_f, was_s = fast_ma.iloc[i - 1], slow_ma.iloc[i - 1]
        if pd.isna(now_f) or pd.isna(now_s) or pd.isna(was_f) or pd.isna(was_s):
            continue
        if not (was_f <= was_s and now_f > now_s):
            continue
        nxt = bars.iloc[i + 1]
        if float(closes.iloc[i]) <= 0 or float(nxt["open"]) <= 0:
            continue
        events.append(
            BounceEvent(
                code=code,
                trigger_date=bars.index[i],
                drop_pct=float(now_f / now_s - 1.0),
                entry_close=float(closes.iloc[i]),
                next_open=float(nxt["open"]),
                next_close=float(nxt["close"]),
            )
        )
    return events


def find_new_high_events(
    code: str, bars: pd.DataFrame, period: int = 20, **_ignored
) -> list[BounceEvent]:
    """Closes above the highest close of the previous *period* sessions."""
    warmup = period + 2
    if len(bars) < warmup + 2:
        return []
    closes = bars["close"].astype(float)
    prior_high = closes.rolling(period).max().shift(1)

    events: list[BounceEvent] = []
    for i in range(warmup, len(bars) - 1):
        level = prior_high.iloc[i]
        if pd.isna(level) or float(closes.iloc[i]) <= float(level):
            continue
        nxt = bars.iloc[i + 1]
        if float(nxt["open"]) <= 0:
            continue
        events.append(
            BounceEvent(
                code=code,
                trigger_date=bars.index[i],
                drop_pct=float(closes.iloc[i] / level - 1.0),
                entry_close=float(closes.iloc[i]),
                next_open=float(nxt["open"]),
                next_close=float(nxt["close"]),
            )
        )
    return events


#: Pattern name -> (finder, report title, description template).
PATTERNS = {
    "drop": (
        find_bounce_events,
        "DROP AND BOUNCE",
        "{down_days} consecutive down sessions falling {drop_pct:.0%} or more in total.",
    ),
    "strong-close": (
        find_strength_events,
        "STRONG CLOSE, NEXT SESSION",
        "a close in the top 30% of the day's range, above the 20-day EMA, "
        "on 1.2x volume.",
    ),
    "volume-spike": (
        find_volume_spike_events,
        "VOLUME SPIKE, NEXT SESSION",
        "volume at least 5x its own 20-day average, whatever the price did.",
    ),
    "ma-cross": (
        find_ma_cross_events,
        "5/20 CROSS, NEXT SESSION",
        "the session the 5-day EMA crossed above the 20-day.",
    ),
    "new-high": (
        find_new_high_events,
        "20-DAY HIGH, NEXT SESSION",
        "a close above the highest close of the previous 20 sessions.",
    ),
}


def format_bounce_study(
    stats: Sequence[BounceStats],
    down_days: int,
    drop_pct: float,
    cost_pct: float,
    limit_down: bool,
    survivorship_note: bool = False,
    per_stock: bool = True,
    pattern: str = "drop",
) -> str:
    width = 100
    lines = ["=" * width]
    if pattern != "drop":
        title = PATTERNS[pattern][1]
    else:
        title = "TWO-DAY DROP, NEXT-DAY BOUNCE" if down_days == 2 else "DROP AND BOUNCE"
    lines += [title.center(width), "=" * width, ""]

    if pattern != "drop":
        lines.append("  Pattern: " + PATTERNS[pattern][2])
    elif limit_down:
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
    if survivorship_note:
        lines.append("")
        lines.append("  Survivorship: the stocks that fell and never came back are the")
        lines.append("  ones most likely to be missing from any ticker list, and they are")
        lines.append("  exactly the counter-examples this pattern needs. Read a positive")
        lines.append("  number here as an upper bound, not as an estimate.")
    lines.append("")

    header = (
        f"  {'code':<8} {'name':<14} {'sessions':>9} {'events':>7}"
        f"{'  |':>3} {'mean':>8} {'median':>8} {'win':>6} {'best':>8} {'worst':>8}"
    )
    if per_stock:
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
    else:
        active = [s for s in stats if s.count]
        lines.append(
            f"  {len(stats)} stocks scanned, {len(active)} produced at least one event."
        )

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

        # Independence check. See _pool_by_date: a market-wide fall makes many
        # stocks qualify on the same day, and those are one observation.
        dated_open = _pool_by_date(stats, cost_pct, to_open=True)
        dated_close = _pool_by_date(stats, cost_pct, to_open=False)
        distinct = int(dated_open["n"])
        if distinct < total_events:
            lines.append("")
            lines.append(
                f"    These {total_events} events happened on {distinct} distinct "
                f"dates (largest single day: {int(dated_open['largest_cluster'])} "
                "stocks at once)."
            )
            lines.append(
                "    A 3-day fall of this size is usually the market, not the company,"
            )
            lines.append(
                "    so same-day events are one observation. Counting each date once:"
            )
            lines.append(
                f"      to next open : mean {dated_open['mean']:+.2%}   "
                f"win {dated_open['win_rate']:.0%}   t {dated_open['t_stat']:+.2f}   "
                f"n={distinct}"
            )
            lines.append(
                f"      to next close: mean {dated_close['mean']:+.2%}   "
                f"win {dated_close['win_rate']:.0%}   t {dated_close['t_stat']:+.2f}   "
                f"n={distinct}"
            )

        lines.append("")
        verdict = _verdict(
            pooled_open, pooled_close, total_events, dated_open, dated_close
        )
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


def _pool_by_date(
    stats: Sequence[BounceStats], cost_pct: float, to_open: bool
) -> dict[str, float]:
    """Pool one observation per trigger date rather than one per stock.

    A 20% fall over three sessions is usually not a company event -- it is the
    market falling, and on those days many stocks qualify at once and bounce
    together. Counting each stock separately then treats one market event as a
    dozen independent ones, which is exactly the assumption a t-statistic
    rests on. Averaging within a date and pooling across dates gives the
    number of independent observations there actually were.
    """
    by_date: dict[Any, list[float]] = {}
    for s in stats:
        for e in s.events:
            value = (e.to_open_pct if to_open else e.to_close_pct) - cost_pct
            key = getattr(e.trigger_date, "date", lambda: e.trigger_date)()
            by_date.setdefault(key, []).append(value)
    if not by_date:
        return {"mean": 0.0, "t_stat": 0.0, "n": 0, "largest_cluster": 0}
    values = [sum(v) / len(v) for v in by_date.values()]
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std_error = (variance / n) ** 0.5
    else:
        std_error = float("inf")
    return {
        "mean": mean,
        "win_rate": sum(1 for v in values if v > 0) / n,
        "t_stat": mean / std_error if std_error else 0.0,
        "n": n,
        "largest_cluster": max(len(v) for v in by_date.values()),
    }


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
    to_open: Mapping[str, float],
    to_close: Mapping[str, float],
    count: int,
    dated_open: Mapping[str, float] | None = None,
    dated_close: Mapping[str, float] | None = None,
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

    # The date-clustered figure outranks the per-event one: if it disagrees,
    # the per-event t was counting one market move many times over.
    dated = dated_open if to_open["mean"] >= to_close["mean"] else dated_close
    if dated and dated.get("n", 0) and dated["n"] < count:
        dated_t = abs(dated.get("t_stat", 0.0))
        if t_stat >= 2.0 > dated_t:
            lines.append("")
            lines.append(
                f"But counting each date once, t falls to {dated.get('t_stat', 0.0):+.2f} "
                f"on {int(dated['n'])} dates. The per-event"
            )
            lines.append(
                "figure above was treating one market-wide fall as many independent"
            )
            lines.append(
                "results. Take the date-clustered number as the real one: this is not"
            )
            lines.append("established.")
        elif dated_t >= 2.0:
            lines.append("")
            lines.append(
                f"It survives date clustering too: t {dated.get('t_stat', 0.0):+.2f} "
                f"across {int(dated['n'])} distinct dates."
            )

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


def _looks_like_codes(values) -> list[str]:
    """Keep only six-digit tickers, in order, without duplicates."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        code = str(value).strip()
        if len(code) == 6 and code.isdigit() and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def _ticker_list_attempts(stock, stamp: str, board: str):
    """Ways of asking pykrx which stocks were listed, best first.

    ``get_market_ticker_list`` is the direct question and the one that has
    been failing: KRX's listing endpoint returns something that is not JSON.
    The whole-market OHLCV snapshot answers the same question from an
    endpoint that demonstrably works -- every stock that traded that day is a
    stock that was listed that day -- so it is worth trying rather than
    reporting an empty list and stopping.
    """
    yield "get_market_ticker_list", lambda: stock.get_market_ticker_list(
        stamp, market=board
    )
    for name in ("get_market_ohlcv_by_ticker", "get_market_cap_by_ticker"):
        fn = getattr(stock, name, None)
        if fn is not None:
            yield name, (lambda fn=fn: list(fn(stamp, board).index))
    yield "get_market_ohlcv(market=)", lambda: list(
        stock.get_market_ohlcv(stamp, market=board).index
    )


def market_codes(
    market: str,
    as_of: "date | None" = None,
    limit: int | None = None,
    broker=None,
) -> tuple[list[str], str]:
    """Every ticker on a board, as of *as_of*.

    Taking the list as of the START of the study window rather than today is
    deliberate. A list of what is listed *now* has already removed every stock
    that fell and never came back -- which for a study of what happens after a
    crash is the single worst bias available. The list is still imperfect
    (bars for a delisted name may not come back at all) but it is honest about
    what it is trying to do.
    """
    from datetime import date as _date
    from datetime import timedelta as _timedelta

    from data import _import_pykrx_stock
    from market.rules import KOSDAQ, KOSPI

    boards = {"kospi": [KOSPI], "kosdaq": [KOSDAQ], "all": [KOSPI, KOSDAQ]}
    wanted = boards.get(market.lower())
    if wanted is None:
        raise ValueError(
            f"unknown market {market!r}; expected one of {sorted(boards)}"
        )
    start = as_of or _date.today()

    out: list[str] = []
    try:
        stock = _import_pykrx_stock()
    except ImportError:
        stock = None
    for board in wanted:
        found: list[str] = []
        if stock is None:
            break
        # A weekend or holiday returns nothing however the question is asked,
        # so step back a few days before concluding the source is broken.
        for offset in range(0, 8):
            stamp = (start - _timedelta(days=offset)).strftime("%Y%m%d")
            for label, attempt in _ticker_list_attempts(stock, stamp, board):
                try:
                    found = _looks_like_codes(attempt())
                except Exception as exc:
                    logger.debug("%s failed for %s on %s: %s", label, board, stamp, exc)
                    continue
                if found:
                    logger.info(
                        "%s: %d tickers from %s on %s", board, len(found), label, stamp
                    )
                    break
            if found:
                break
        if not found:
            logger.warning("no ticker list for %s near %s", board, start)
        out.extend(found)

    source = "pykrx, as of the start of the window"
    if not out and broker is not None:
        # Every KRX market-wide endpoint is down. The broker still knows what
        # is listed -- but only what is listed *now*, so the survivorship
        # caveat gets stronger rather than softer.
        source = "the broker's CURRENT listing"
        for board in wanted:
            name = "kospi" if board.upper().startswith("KOSPI") else "kosdaq"
            try:
                out.extend(broker.get_market_codes(name))
            except Exception as exc:
                logger.warning("broker ticker list for %s failed: %s", name, exc)

    seen: set[str] = set()
    unique = [c for c in out if not (c in seen or seen.add(c))]
    return (unique[:limit] if limit else unique), source


def run_bounce_study(
    universe: Mapping[str, Mapping[str, Any]],
    market_data,
    months: int = 24,
    down_days: int = 2,
    drop_pct: float = 0.15,
    limit_down: bool = False,
    progress: "Callable[[int, int], None] | None" = None,
    pattern: str = "drop",
) -> list[BounceStats]:
    from datetime import datetime

    from data import months_to_start
    from market.calendar import KST
    from market.rules import KOSPI

    end = datetime.now(KST)
    start = months_to_start(months, end)

    out: list[BounceStats] = []
    total = len(universe)
    for done, (code, cfg) in enumerate(universe.items(), start=1):
        code = str(code)
        if progress is not None:
            progress(done, total)
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
        stats.baseline = overnight_returns(barset.bars)
        if barset.synthetic:
            logger.warning("%s: synthetic bars - excluded from the study", code)
        else:
            finder = PATTERNS[pattern][0]
            if pattern == "drop":
                stats.events = finder(
                    code, barset.bars, down_days, drop_pct, limit_down
                )
            else:
                stats.events = finder(code, barset.bars)
        out.append(stats)
    return out


# ---------------------------------------------------------------------------
# Which stocks, if any, are worth selecting
# ---------------------------------------------------------------------------


def overnight_returns(bars: pd.DataFrame) -> "pd.Series":
    """Close-to-next-open return for every session, indexed by session date."""
    if len(bars) < 2:
        return pd.Series(dtype="float64")
    closes = bars["close"].astype(float)
    opens = bars["open"].astype(float)
    values = (opens.shift(-1) / closes - 1.0).iloc[:-1]
    values.index = [
        getattr(ts, "date", lambda: ts)() for ts in bars.index[:-1]
    ]
    return values.dropna()


def _split_date(stats: Sequence[BounceStats]):
    """The date that divides the events into two halves of equal count."""
    dates = sorted(
        getattr(e.trigger_date, "date", lambda: e.trigger_date)()
        for s in stats
        for e in s.events
    )
    return dates[len(dates) // 2] if dates else None


def _half(stats: BounceStats, split, early: bool) -> list[BounceEvent]:
    out = []
    for e in stats.events:
        day = getattr(e.trigger_date, "date", lambda: e.trigger_date)()
        if (day < split) if early else (day >= split):
            out.append(e)
    return out


def _mean_to_open(events: Sequence[BounceEvent], cost_pct: float) -> float:
    if not events:
        return 0.0
    return sum(e.to_open_pct - cost_pct for e in events) / len(events)


def format_selection_study(
    stats: Sequence[BounceStats],
    cost_pct: float,
    top_n: int = 10,
    min_events: int = 5,
) -> str:
    """Pick the best stocks on the first half; see how they do on the second.

    Ranking 300 stocks by past return and trading the top ten will always
    produce a good-looking list, because with 300 tries something scores well
    by chance alone. The only way to know whether the ranking meant anything
    is to choose from data the test never sees -- so selection happens on the
    earlier half of the history and the verdict comes from the later half,
    compared against what every stock did over that same later period.

    If the chosen ten beat the field out of sample, stock selection works
    here. If they land on the field average, the ranking was noise and the
    right number of stocks to select is all of them.
    """
    width = 100
    lines = ["=" * width, "WHICH STOCKS, IF ANY".center(width), "=" * width, ""]

    split = _split_date(stats)
    if split is None:
        lines += ["  No events at all, so nothing to rank.", "=" * width]
        return "\n".join(lines)

    scored = []
    for s in stats:
        early = _half(s, split, early=True)
        late = _half(s, split, early=False)
        if len(early) >= min_events:
            scored.append((s, early, late, _mean_to_open(early, cost_pct)))

    lines.append(f"  Selection window : events before {split}")
    lines.append(f"  Test window      : events from {split}")
    lines.append(
        f"  Eligible         : {len(scored)} stocks with at least {min_events} "
        "events in the selection window"
    )
    lines.append("")

    if not scored:
        lines += [
            f"  No stock had {min_events} events before {split}, so there is nothing",
            "  to select on. A longer history or a looser rule would be needed.",
            "=" * width,
        ]
        return "\n".join(lines)

    scored.sort(key=lambda row: row[3], reverse=True)
    chosen = scored[:top_n]

    lines.append(f"  -- Top {len(chosen)} by the selection window " + "-" * (width - 40))
    lines.append(
        f"  {'code':<8} {'name':<14} {'sel n':>6} {'sel mean':>10}"
        f" {'test n':>7} {'test mean':>10}"
    )
    lines.append("  " + "-" * (width - 4))
    for s, early, late, score in chosen:
        test = _mean_to_open(late, cost_pct) if late else float("nan")
        shown = f"{test:+10.2%}" if late else f"{'-':>10}"
        lines.append(
            f"  {s.code:<8} {(s.name or '')[:13]:<14} {len(early):>6d} "
            f"{score:>+10.2%} {len(late):>7d} {shown}"
        )

    chosen_late = [e for _, _, late, _ in chosen for e in late]
    field_late = [e for s in stats for e in _half(s, split, early=False)]
    chosen_mean = _mean_to_open(chosen_late, cost_pct)
    field_mean = _mean_to_open(field_late, cost_pct)

    # The control. A stock that rose all through the test window pays any rule
    # that buys it, so the pattern has to beat simply holding the same stocks
    # over the same days -- gross, since holding does not pay a round trip per
    # session.
    held = [
        value
        for s, _, _, _ in chosen
        if s.baseline is not None
        for day, value in s.baseline.items()
        if day >= split
    ]
    hold_mean = sum(held) / len(held) if held else 0.0

    # And the same clustering question as everywhere else.
    dates: dict[Any, list[float]] = {}
    for e in chosen_late:
        key = getattr(e.trigger_date, "date", lambda: e.trigger_date)()
        dates.setdefault(key, []).append(e.to_open_pct - cost_pct)
    per_date = [sum(v) / len(v) for v in dates.values()]
    if len(per_date) > 1:
        mean = sum(per_date) / len(per_date)
        variance = sum((v - mean) ** 2 for v in per_date) / (len(per_date) - 1)
        std_error = (variance / len(per_date)) ** 0.5
        dated_t = mean / std_error if std_error else 0.0
    else:
        dated_t = 0.0

    lines += ["", "  " + "-" * (width - 4)]
    lines.append("  Out of sample, in the test window:")
    lines.append(
        f"    the {len(chosen)} chosen stocks : {chosen_mean:+.2%} "
        f"over {len(chosen_late)} events on {len(per_date)} dates  (t {dated_t:+.2f})"
    )
    lines.append(
        f"    every stock            : {field_mean:+.2%} "
        f"over {len(field_late)} events"
    )
    lines.append(
        f"    holding those 10       : {hold_mean:+.2%} per session, "
        f"every session, no costs"
    )
    lines.append("")

    edge = chosen_mean - field_mean
    over_hold = chosen_mean - hold_mean
    if not chosen_late:
        lines.append("  The chosen stocks produced no events in the test window, so the")
        lines.append("  selection cannot be checked at all.")
    elif edge <= 0:
        lines.append(
            f"  Selecting made it {abs(edge):.2%} WORSE than taking the whole field."
        )
        lines.append(
            "  The ranking was noise: last period's best stocks are not next"
        )
        lines.append(
            "  period's. Picking stocks by past performance does not work here."
        )
    elif over_hold <= 0:
        lines.append(
            f"  The pattern returned {chosen_mean:+.2%} on these stocks while simply"
        )
        lines.append(
            f"  holding them overnight returned {hold_mean:+.2%} a session. The rule adds"
        )
        lines.append(
            "  nothing: it picked stocks that were going up, and any rule that bought"
        )
        lines.append("  them would have looked the same.")
    elif edge < cost_pct:
        lines.append(
            f"  Selecting gained {edge:+.2%} over the field -- smaller than the"
        )
        lines.append(
            f"  {cost_pct:.2%} it costs to trade at all. Not worth acting on."
        )
    elif abs(dated_t) < 2.0:
        lines.append(
            f"  Better than the field ({edge:+.2%}) and better than holding "
            f"({over_hold:+.2%}),"
        )
        lines.append(
            f"  but t is {dated_t:+.2f} across {len(per_date)} dates -- inside what chance"
        )
        lines.append(
            "  produces. One split of one board is not enough to act on."
        )
    else:
        lines.append(
            f"  Beats the field by {edge:+.2%}, beats holding the same stocks by"
        )
        lines.append(
            f"  {over_hold:+.2%}, and survives date clustering at t {dated_t:+.2f}."
        )
        lines.append(
            "  Rerun on KOSPI and on a different split before trusting it."
        )
    lines.append("=" * width)
    return "\n".join(lines)
