# KRX 8-Stock Trading Bot

A paper-first trading bot for eight KRX-listed stocks, built on the **Kiwoom
REST API** and Python 3.10+.

**Paper trading is the default and the fallback.** Live trading requires three
independent confirmations; miss any one and the run is demoted to paper with
the reason printed. See [Safety rules](#safety-rules).

> **Not investment advice.** This is engineering scaffolding for studying a set
> of rules. Backtest numbers are not a forecast, and the defaults have not been
> tuned for profitability.

---

## Universe

| Code | Name | Market | Timeframe | Strategy | Theme |
|---|---|---|---|---|---|
| 009830 | 한화솔루션 | KOSPI | daily | EMA trend (50/200) | power_energy |
| 002990 | 금호건설 | KOSPI | daily | EMA trend (50/200) | construction |
| 093370 | 후성 | KOSPI | daily | EMA trend (50/200) | battery_materials |
| 006340 | 대원전선 | KOSPI | daily | EMA trend (50/200) | power_energy |
| 460930 | 현대힘스 | KOSDAQ | 60 min | Volume breakout (20) | shipbuilding |
| 101730 | 위메이드맥스 | KOSDAQ | 60 min | Volume breakout (20) | game |
| 228340 | 동양파일 | KOSDAQ | 60 min | Volume breakout (20) | construction |
| 439960 | 코스모로보틱스 | KOSDAQ | 60 min | Volume breakout (**10**) | robotics |

Names and market tags in `config.yaml` are a starting point — **verify them**
with `python main.py profile`, which reads the official listing data. A wrong
`market` tag mis-prices the transaction tax.

### Two entries that need explaining

**439960 코스모로보틱스 listed on 2026-05-11.** With roughly three months of
history it cannot warm up a 200-period lookback, so it runs a shortened
10-period breakout instead. `profile` reports bar counts against each
strategy's warm-up requirement, and the bot holds rather than trading a
half-computed indicator. Revisit the parameters once it has a year behind it.

**Mean reversion ships disabled.** It is implemented and tested, but it is not
assigned to any stock. On an index, −1.5σ tends to revert; on an individual
small cap it can be the start of a repricing, and unlike an index a single
stock can be suspended, diluted or delisted. Turn it on per-stock in
`config.yaml` if you want it.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # optional: works without it, on synthetic data
python main.py profile     # verify names, liquidity, correlation
python main.py backtest --months 6
python main.py paper --dry-run
```

Nothing here needs credentials to run. Without a Kiwoom app key the bot falls
back to a simulated broker that prints intended orders and sends nothing, and
market data falls back to a deterministic synthetic generator (loudly flagged
as `SYNTHETIC DATA` wherever it is used).

---

## Commands

```bash
python main.py profile                    # measure the universe, suggest strategies
python main.py profile --refresh-calendar # derive KRX holidays from pykrx
python main.py backtest --months 6        # next-bar fills, costs included
python main.py paper --dry-run            # one cycle, print orders, send nothing
python main.py paper --once               # one cycle against the mock account
python main.py live --once                # needs all three confirmations
python main.py status                     # state, positions, drawdown, capacity
python main.py stop --close-all           # cancel orders, flatten, persist STOPPED
python main.py resume                     # clear STOPPED

python report.py morning --dry-run        # pre-session briefing (Telegram preview)
python report.py evening --dry-run        # end-of-day wrap
```

`report.py` is dry-run by default; pass `--send` to actually deliver.

---

## Safety rules

| # | Rule | Implementation | Verified by |
|---|---|---|---|
| 1 | Paper is the default. With no environment at all, only `mockapi.kiwoom.com` is reachable | `settings.resolve_mode` | `test_live_requires_triple_confirm.py::test_empty_environment_is_paper` |
| 2 | Live requires **all three**: `KIWOOM_PAPER=false`, `KIWOOM_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY`, an explicit live request on the command line | `settings.resolve_mode`, `settings.load_credentials` | the full 8-row truth table |
| 3 | Live start prints a red banner, balance, worst-case loss per trade *and* per portfolio, then a 10-second countdown. Every log line carries the mode | `main.print_live_banner`, `settings.setup_logging` | manual; `mode=` on every line of `logs/bot.log` |
| 4 | Risk per trade = 1% of **current** equity. Quantity = `(equity × 0.01) ÷ stop distance`. Stop distance ≤ 0 → skip the order | `risk/manager.py::position_size` | `test_risk.py` |
| 5 | 10% drawdown from peak → block new orders, cancel working orders, flatten, persist `STOPPED` | `risk/manager.py::RiskManager.trip_kill_switch` | `test_risk.py::test_kill_switch_stops_cancels_and_flattens` |
| 6 | `STOPPED` persists across restarts until a human runs `resume` | `portfolio.StateStore` + `state.json` | `test_risk.py::test_stopped_state_survives_a_restart` |
| 7 | Before every order: tradability, session phase, minimum quantity, duplicate orders, available cash, **daily price limit** | `risk/manager.py::pre_trade_checks` | `test_risk.py` (one test per check) |
| 8 | KRX session only — 09:00–15:30 KST, never during a call auction | `market/calendar.py` | `test_calendar.py` |
| 9 | App keys come from `.env` only and never reach logs, exception messages or reports | `settings.Secret`, `SecretFilter`, `install_exception_masking` | canary test; secrets render as `***REDACTED***` |
| 10 | Backtest fills start on the bar **after** the signal bar | `backtest.Backtester` | `test_no_lookahead.py` |
| 11 | Commission, transaction tax and slippage live in `config.yaml` and print with every result | `market/rules.py::TradingCosts` | `test_krx_rules.py`, `test_no_lookahead.py` |
| 12 | Telegram is dry-run by default; a missing token gives a preview, not an error | `report.send_telegram` | `python report.py morning --dry-run` |

### The live-trading gate

```
KIWOOM_PAPER=false                              (environment)
KIWOOM_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY     (environment, exact match)
`live` sub-command, or --live on `paper`        (command line)
```

Only all three together open the live endpoint. The `live` sub-command *is* the
explicit command-line request, so `python main.py live --once` satisfies the
third condition — and with the environment variables absent it still demotes:

```
==============================================================================
  LIVE TRADING WAS REQUESTED BUT DEMOTED TO PAPER
    - KIWOOM_PAPER is not 'false' (effective value: 'true', default is 'true')
    - KIWOOM_LIVE_CONFIRM is not set
  Running against https://mockapi.kiwoom.com
==============================================================================
```

`resolve_mode` is a pure function with no I/O, and it is the only place that can
produce a live base URL. That is what makes the truth-table test exhaustive
rather than indicative.

**Kiwoom's split keys buy one more layer.** Kiwoom issues *separate* app keys
for the mock and live environments, so `load_credentials` reads only the set
matching the resolved mode. On a paper run the live key is never pulled out of
the environment at all — a mis-assembled URL still cannot authenticate against
a real account, because the credential is not in memory. `KiwoomBroker`
additionally refuses to start if the credential set and the mode disagree.

### Position sizing

The hard stop sits exactly one ATR(14) from entry, and quantity is the risk
budget divided by that distance:

```
stop_distance = ATR(14)
quantity      = (equity × 1%) ÷ stop_distance      (whole shares, rounded down)
```

A 1-ATR adverse move therefore costs 1% of equity on every stock: a quiet name
gets size, a violent one does not, and the money at risk never changes. Three
caps can only ever *reduce* it: `max_position_notional_pct` (25%), available
cash, and the remaining portfolio risk budget.

### Portfolio caps

Eight stocks at 1% each would put 8% at risk simultaneously — the US-market
version of this bot ran five broad instruments, where that was not a concern.
Here two caps apply:

* **6% total open risk**, measured against each position's *effective* stop, so
  a trailing stop that has ratcheted past entry frees its budget back up;
* **6 open positions** at once.

### Theme filter

Two stocks in one theme are one thesis sized twice. Only one position per theme
is allowed:

* `construction` — 002990 금호건설, 228340 동양파일
* `power_energy` — 006340 대원전선, 009830 한화솔루션

Exits are never blocked. `profile` prints the return-correlation matrix so the
groups can be checked against how the stocks actually move.

### Long-only

Retail short selling of KRX equities is not practically available, and futures
are out of scope — which also rules out inverse ETFs, since those hold index
futures internally. A short signal therefore **closes a position but never
opens one**. `risk.long_only` in `config.yaml` controls it, and it propagates
to every strategy through `allow_short`.

---

## KRX market rules

Three things differ from a US venue, and each one silently breaks a
US-shaped assumption:

**Tick size.** KRX quotes on a price-dependent grid (1 / 5 / 10 / 50 / 100 /
500 / 1000 KRW, with KOSDAQ topping out at 100). An off-grid order is rejected,
so every stop is snapped to the grid — **rounding up** for a long stop, because
rounding down would place it further away than the 1% budget allows.

**Daily price limit (±30%).** At limit-up there are no sellers and at
limit-down no buyers, so orders there would not fill. Both the live loop and
the backtest skip rather than pretending they traded.

**Asymmetric costs.** Buying pays commission; **selling pays commission plus
transaction tax.** A round trip therefore costs materially more than the US
model would suggest, which matters most for the 60-minute breakout strategies.
Rates live in `config.yaml` — *verify them*, they have been cut in stages.

**Session.** 09:00–15:30 KST continuous trading, with call auctions at
08:30–09:00 and 15:20–15:30. Market orders are refused during auctions, where
there is no continuous book and the execution price is unpredictable.

**Holidays.** Korea's calendar has lunar holidays, substitute holidays and
ad-hoc closures that cannot be derived from a rule. Supply them in
`config.yaml` under `krx.holidays`, or generate the list:

```bash
python main.py profile --refresh-calendar   # needs pykrx + network
```

With no list configured the calendar falls back to weekends-only **and says
so** rather than silently treating a holiday as a trading day.

---

## Market data

Every request resolves in this order:

1. **CSV cache** under `data/cache/`;
2. **live source** — `pykrx` for daily bars, the **Kiwoom chart API** for
   intraday bars (pykrx does not serve minute data);
3. **synthetic generator** — deterministic, seeded per stock.

Step 3 lets the whole pipeline run with no credentials and no network. Anything
that consumed synthetic bars is stamped `SYNTHETIC DATA`.

### Warm-up history is fetched separately

A 200-day EMA needs about ten calendar months of history. Without care,
`backtest --months 6` would spend its entire window warming up and trade
nothing. `data.warmup_start` therefore extends the fetch backwards by each
strategy's warm-up requirement, so the requested window is actually tradable.
The backtest header shows both numbers:

```
009830 한화솔루션        354 bars  [synthetic]  warmup 202 -> 152 tradable
460930 현대힘스         1001 bars  [synthetic]  warmup  22 -> 979 tradable
```

---

## Backtesting

Event-driven across all eight stocks on one shared equity curve, so the theme
filter, the portfolio risk cap and the drawdown kill switch behave as they do
live. Signals are evaluated on bar `i` and executed at the **open of bar
`i+1`**, with slippage against the order.

The no-look-ahead guarantee is structural:

1. `Strategy.evaluate` receives `bars.iloc[:i+1]` — a future bar is unreachable;
2. an actionable signal schedules a pending order at `i+1`, and every fill
   records both indices;
3. the engine asserts `fill_index == signal_index + 1` before returning, and
   `fill_delay_bars = 0` is rejected at construction.

`tests/test_no_lookahead.py` also rewrites the tail of a series and checks that
no earlier signal or fill moves.

The drawdown kill switch applies during backtests too: at 10% below peak,
trading halts for the rest of the period and the summary says so.

---

## Continuous operation

`python main.py paper` wakes every 30 seconds and checks each stock on its own
cadence (hourly / daily, configurable). Each cycle:

1. read the account (retried with jittered exponential backoff);
2. update peak equity and evaluate the drawdown guard;
3. bail out immediately if the state is `STOPPED`;
4. skip everything outside continuous trading;
5. per stock: ratchet the trailing stop, **check the hard stop**, evaluate the
   signal, size against remaining portfolio risk, run pre-trade checks, submit.

Step 5's stop check matters: **Kiwoom has no bracket order**, so the loop
enforces the hard stop itself rather than leaving it resting at the broker.
Kiwoom also throttles TR calls, so every request passes through a rate limiter
before the retry wrapper.

---

## Tests

```bash
pytest -q
```

214 tests, no network and no credentials required. `tests/conftest.py` scrubs
`KIWOOM_*` from the environment before every test, so a real `.env` can never
turn a test run into a live-trading attempt, and the broker double raises if
anything tries to submit an order.

---

## Files

```
main.py                   CLI, live banner, continuous loop
config.yaml               universe, themes, risk limits, KRX rules, costs
.env.example              credential template (mock and live keys kept apart)
settings.py               .env loading, secret masking, resolve_mode  ← safety core
data.py                   bars: cache → pykrx/Kiwoom → synthetic; warm-up windows
indicators.py             SMA / EMA / stddev / ATR / rolling high-low
broker.py                 BrokerBase, KiwoomBroker, DryRunBroker, rate limit + retry
portfolio.py              positions, state.json, trades.csv, daily_pnl.csv
backtest.py               next-bar-fill engine, KRX cost model, metrics
universe_profile.py       volatility / liquidity / correlation, strategy suggestions
report.py                 morning & evening reports, Telegram dry-run
market/
  calendar.py             session phases, business days, auction safety
  rules.py                tick ladder, ±30% limits, asymmetric costs
strategies/               base, trend_following, breakout, mean_reversion
risk/manager.py           sizing, hard stops, kill switch, portfolio caps, checks
tests/                    risk, triple-confirmation, no-lookahead, KRX rules, calendar
```

Runtime artefacts (git-ignored): `trades.csv`, `daily_pnl.csv`, `state.json`,
`logs/bot.log`, `data/cache/`.

### Output files

`trades.csv` — one row per closed trade:

```
timestamp,symbol,side,entry_price,exit_price,pnl,qty,
strategy,entry_time,fees,slippage,return_pct,exit_reason,mode
```

`daily_pnl.csv` — one row per day (repeated runs update that day's row):

```
date,starting_equity,ending_equity,realized_pnl,unrealized_pnl,
trades,max_drawdown_pct,mode
```

---

## Before trading real money

Two categories of constant in this repo were written from documentation rather
than from a live account, and both are config-overridable so you can correct
them without touching code:

* **Kiwoom endpoint paths and `api-id` codes** (`broker.py` defaults,
  overridable under `kiwoom.endpoints` / `kiwoom.api_ids`) — check them against
  <https://openapi.kiwoom.com/> and the
  [official examples](https://github.com/Kiwoom-Securities/Kiwoom-REST-API);
* **tick ladder and tax rates** (`config.yaml`) — check them against the
  current KRX rulebook.

Run the mock account long enough to see fills before changing any of the three
live confirmations.
