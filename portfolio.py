"""Position bookkeeping, persistent state and CSV trade records.

Three artefacts are maintained here:

``state.json``      run status (RUNNING / STOPPED), peak equity for the
                    drawdown guard, and the open positions the bot believes
                    it holds.  The STOPPED flag survives restarts, which is
                    what makes safety rule 6 ("stays stopped until a human
                    resumes") actually hold.
``trades.csv``      one row per closed trade.
``daily_pnl.csv``   one row per calendar day.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable

# market.calendar imports only the standard library, so this cannot cycle.
from market.calendar import KST

logger = logging.getLogger("bot.portfolio")

LONG = "LONG"
SHORT = "SHORT"

STATUS_RUNNING = "RUNNING"
STATUS_STOPPED = "STOPPED"

TRADE_COLUMNS = (
    "timestamp",
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "pnl",
    "qty",
    # Context columns beyond the required set.
    "strategy",
    "entry_time",
    "fees",
    "slippage",
    "return_pct",
    "exit_reason",
    "mode",
)

DAILY_COLUMNS = (
    "date",
    "starting_equity",
    "ending_equity",
    "realized_pnl",
    "unrealized_pnl",
    "trades",
    "max_drawdown_pct",
    "mode",
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------


@dataclass
class Position:
    """An open position as the bot understands it."""

    symbol: str
    side: str
    qty: float
    entry_price: float
    stop_price: float
    stop_distance: float
    strategy: str = ""
    entry_time: str = ""
    trail_stop: float | None = None
    take_profit: float | None = None
    highest_price: float | None = None
    lowest_price: float | None = None
    #: Set once this position's gain crosses risk.near_limit_hold_pct (a
    #: 상한가 candidate) -- from then on it skips the same-day force exit and
    #: profit-lock trail and is sold the next morning instead. See
    #: _late_exit_reason's neighbor in main.py for the matching cycle logic.
    near_limit_hold: bool = False

    @property
    def is_long(self) -> bool:
        return self.side == LONG

    @property
    def is_short(self) -> bool:
        return self.side == SHORT

    @property
    def direction(self) -> int:
        return 1 if self.is_long else -1

    def unrealized(self, price: float) -> float:
        return (price - self.entry_price) * self.qty * self.direction

    def entry_date(self) -> "date | None":
        """The KRX session this position was opened in, if it is known.

        Used to tell an overnight hold that has reached its planned morning
        exit from one that was opened minutes ago, after that time of day had
        already passed. ``entry_time`` is free-form text on purpose -- it has
        been written by several code paths -- so this parses defensively and
        answers None rather than guessing a date it cannot read.
        """
        raw = (self.entry_time or "").strip()
        if not raw:
            return None
        try:
            moment = datetime.fromisoformat(raw)
        except ValueError:
            try:
                return date.fromisoformat(raw[:10])
            except ValueError:
                return None
        if moment.tzinfo is not None:
            moment = moment.astimezone(KST)
        return moment.date()

    def entry_time_of_day(self) -> "time | None":
        """The KST wall-clock time this position was opened, if known.

        Same defensive parsing as entry_date() (entry_time is free-form
        text, written by several code paths). Used to lock a position to
        whichever exit tier was in force at entry -- e.g. a 11:00-15:00
        window with different stop/lock percentages (2026-08-28, user
        request) -- so a still-open position never switches tiers mid-trade
        just because the wall clock moved on.
        """
        raw = (self.entry_time or "").strip()
        if not raw:
            return None
        try:
            moment = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if moment.tzinfo is not None:
            moment = moment.astimezone(KST)
        return moment.time()

    def effective_stop(self) -> float:
        """The stop actually in force: the tighter of hard stop and trail."""
        if self.trail_stop is None:
            return self.stop_price
        return max(self.stop_price, self.trail_stop) if self.is_long else min(
            self.stop_price, self.trail_stop
        )

    def stop_hit(self, low: float, high: float) -> bool:
        stop = self.effective_stop()
        return low <= stop if self.is_long else high >= stop

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------
# Persistent state
# --------------------------------------------------------------------------


@dataclass
class BotState:
    status: str = STATUS_RUNNING
    peak_equity: float = 0.0
    last_equity: float = 0.0
    stopped_reason: str = ""
    stopped_at: str = ""
    mode: str = "PAPER"
    last_run_at: str = ""
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    day_start_equity: float = 0.0
    day_start_date: str = ""
    #: code -> consecutive losing exits today. A profit exit clears the
    #: entry; cleared for every code at the same day-rollover as
    #: day_start_equity/day_start_date (see mark_equity()).
    symbol_loss_streak: dict[str, int] = field(default_factory=dict)

    @property
    def stopped(self) -> bool:
        return self.status == STATUS_STOPPED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BotState":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class StateStore:
    """Reads and writes ``state.json``."""

    def __init__(self, path: str | Path = "state.json") -> None:
        self.path = Path(path)

    def load(self) -> BotState:
        if not self.path.exists():
            return BotState()
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return BotState.from_dict(json.load(fh))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            # A corrupt state file must not silently reset the STOPPED flag.
            logger.error(
                "state file %s is unreadable (%s); refusing to start in RUNNING",
                self.path,
                exc,
            )
            return BotState(
                status=STATUS_STOPPED,
                stopped_reason=f"state file unreadable: {exc}",
                stopped_at=_iso(utcnow()),
            )

    def save(self, state: BotState) -> None:
        _atomic_write(self.path, json.dumps(state.to_dict(), indent=2, sort_keys=True))


# --------------------------------------------------------------------------
# CSV records
# --------------------------------------------------------------------------


def _append_row(path: Path, columns: Iterable[str], row: dict[str, Any]) -> None:
    columns = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in columns})


@dataclass
class TradeRecord:
    timestamp: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    qty: float
    strategy: str = ""
    entry_time: str = ""
    fees: float = 0.0
    slippage: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = ""
    mode: str = "PAPER"

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


class TradeLog:
    """Appends every completed round trip to ``trades.csv``."""

    def __init__(self, path: str | Path = "trades.csv") -> None:
        self.path = Path(path)

    def record(self, trade: TradeRecord) -> None:
        _append_row(self.path, TRADE_COLUMNS, trade.to_row())

    def read_all(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def rows_for_date(self, day: date) -> list[dict[str, str]]:
        target = day.isoformat()
        return [r for r in self.read_all() if str(r.get("timestamp", "")).startswith(target)]


class DailyPnlLog:
    """Maintains one summary row per day in ``daily_pnl.csv``.

    The loop can call this many times a day, so a repeated date *replaces* the
    existing row rather than appending a duplicate.
    """

    def __init__(self, path: str | Path = "daily_pnl.csv") -> None:
        self.path = Path(path)

    def record(
        self,
        day: date,
        starting_equity: float,
        ending_equity: float,
        realized_pnl: float,
        unrealized_pnl: float,
        trades: int,
        max_drawdown_pct: float,
        mode: str = "PAPER",
    ) -> None:
        row = {
            "date": day.isoformat(),
            "starting_equity": round(starting_equity, 2),
            "ending_equity": round(ending_equity, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "trades": trades,
            "max_drawdown_pct": round(max_drawdown_pct, 6),
            "mode": mode,
        }
        existing = self.read_all()
        if not any(r.get("date") == row["date"] for r in existing):
            _append_row(self.path, DAILY_COLUMNS, row)
            return

        merged = [r if r.get("date") != row["date"] else row for r in existing]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(DAILY_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for entry in merged:
            writer.writerow({c: entry.get(c, "") for c in DAILY_COLUMNS})
        _atomic_write(self.path, buffer.getvalue())

    def read_all(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# Portfolio facade
# --------------------------------------------------------------------------


class Portfolio:
    """Ties state, positions and the two CSV records together."""

    def __init__(
        self,
        state_path: str | Path = "state.json",
        trades_path: str | Path = "trades.csv",
        daily_path: str | Path = "daily_pnl.csv",
        mode_label: str = "PAPER",
    ) -> None:
        self.store = StateStore(state_path)
        self.trades = TradeLog(trades_path)
        self.daily = DailyPnlLog(daily_path)
        self.mode_label = mode_label
        self.state = self.store.load()
        # The mode-mismatch equity reset (see mark_equity()) is deliberately
        # NOT done here. Constructing a Portfolio happens for every command,
        # including read-only ones (status/resume/stop/check) that build a
        # dry-run broker purely to stay safe and never call mark_equity() --
        # 2026-08-28 (live incident): those were resolving a "PAPER-SIM"
        # label on every invocation and wiping peak_equity/day_start_equity
        # (the drawdown kill switch's and the daily profit lock's reference
        # points) each time, even though nothing comparable was ever being
        # recorded to replace them. Only a run that is about to record a
        # real new equity figure needs to -- or is able to correctly --
        # decide whether the stored history is comparable to it.

    # -- positions ---------------------------------------------------------

    def positions(self) -> dict[str, Position]:
        return {s: Position.from_dict(d) for s, d in self.state.positions.items()}

    def get(self, symbol: str) -> Position | None:
        data = self.state.positions.get(symbol)
        return Position.from_dict(data) if data else None

    def open_position(self, position: Position) -> None:
        if not position.entry_time:
            position.entry_time = _iso(utcnow())
        self.state.positions[position.symbol] = position.to_dict()
        self.save()

    def update_position(self, position: Position) -> None:
        self.state.positions[position.symbol] = position.to_dict()
        self.save()

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        fees: float = 0.0,
        slippage: float = 0.0,
        exit_reason: str = "",
        when: datetime | None = None,
    ) -> TradeRecord | None:
        position = self.get(symbol)
        if position is None:
            return None
        gross = position.unrealized(exit_price)
        pnl = gross - fees - slippage
        cost_basis = abs(position.entry_price * position.qty)
        trade = TradeRecord(
            timestamp=_iso(when or utcnow()),
            symbol=symbol,
            side=position.side,
            entry_price=round(position.entry_price, 6),
            exit_price=round(exit_price, 6),
            pnl=round(pnl, 6),
            qty=position.qty,
            strategy=position.strategy,
            entry_time=position.entry_time,
            fees=round(fees, 6),
            slippage=round(slippage, 6),
            return_pct=round(pnl / cost_basis, 6) if cost_basis else 0.0,
            exit_reason=exit_reason,
            mode=self.mode_label,
        )
        self.trades.record(trade)
        self.state.positions.pop(symbol, None)
        self.save()
        return trade

    def record_symbol_result(self, code: str, is_profit: bool) -> int:
        """Update `code`'s same-day consecutive-loss streak; returns the new
        count. A profit exit clears it to 0; a loss increments it.

        User-reported (2026-08-31): several symbols got re-entered 2-4 times
        in one day, with later re-entries losing more often and in larger
        size than the first ("이런식의 매매는 의미없어") -- nothing stopped
        the bot from repeatedly chasing the same symbol back in after it
        kept proving it wasn't cooperating today. See
        main._symbol_loss_streak_reason, which reads this to block further
        entries into just that symbol once the streak is too long.
        """
        if is_profit:
            self.state.symbol_loss_streak.pop(code, None)
            streak = 0
        else:
            streak = self.state.symbol_loss_streak.get(code, 0) + 1
            self.state.symbol_loss_streak[code] = streak
        self.save()
        return streak

    # -- equity / status ---------------------------------------------------

    def mark_equity(self, equity: float) -> None:
        # Equity figures are mode-specific. A dry run reports the configured
        # starting_equity (10,000,000), and carrying that peak into a live
        # account worth 573,000 reads as a 94% drawdown -- which fired the
        # kill switch and liquidated the book on the first live run. Reset
        # the equity history whenever the recorded mode changes; the STOPPED
        # flag and the positions survive, only the numbers that are not
        # comparable go. This only runs here (not at construction -- see the
        # __init__ comment) because this is the one place a run is actually
        # about to make the stored numbers stale or fresh.
        previous = self.state.mode
        if previous and previous != self.mode_label:
            logger.warning(
                "mode changed %s -> %s; discarding equity history "
                "(peak %.0f) because the two are not comparable",
                previous,
                self.mode_label,
                self.state.peak_equity,
            )
            self.state.peak_equity = 0.0
            self.state.last_equity = 0.0
            self.state.day_start_equity = 0.0
            self.state.day_start_date = ""
        self.state.mode = self.mode_label

        self.state.last_equity = equity
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        today = utcnow().date().isoformat()
        if self.state.day_start_date != today:
            self.state.day_start_date = today
            self.state.day_start_equity = equity
            self.state.symbol_loss_streak = {}
        self.state.last_run_at = _iso(utcnow())
        self.save()

    def drawdown_pct(self) -> float:
        peak = self.state.peak_equity
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - self.state.last_equity) / peak)

    def stop(self, reason: str) -> None:
        self.state.status = STATUS_STOPPED
        self.state.stopped_reason = reason
        self.state.stopped_at = _iso(utcnow())
        self.save()

    def resume(self) -> None:
        self.state.status = STATUS_RUNNING
        self.state.stopped_reason = ""
        self.state.stopped_at = ""
        # Re-anchor the peak so the guard measures drawdown from here on.
        self.state.peak_equity = max(self.state.last_equity, 0.0)
        self.save()

    @property
    def stopped(self) -> bool:
        return self.state.stopped

    def save(self) -> None:
        # state.mode is set only by mark_equity() -- see its comment. A save()
        # from any other path (resume/stop/open_position/...) must not stamp
        # this run's label over whatever the last real trading run recorded.
        self.store.save(self.state)

    # -- daily roll-up -----------------------------------------------------

    def record_day(self, equity: float, unrealized: float = 0.0, day: date | None = None) -> None:
        day = day or utcnow().date()
        rows = self.trades.rows_for_date(day)
        realized = sum(float(r.get("pnl") or 0.0) for r in rows)
        start = self.state.day_start_equity or equity
        self.daily.record(
            day=day,
            starting_equity=start,
            ending_equity=equity,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            trades=len(rows),
            max_drawdown_pct=self.drawdown_pct(),
            mode=self.mode_label,
        )
