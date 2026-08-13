# Alpaca 5-Asset Trading Bot

A paper-first trading bot for five assets on the Alpaca API, built with the
official [`alpaca-py`](https://github.com/alpacahq/alpaca-py) SDK and Python
3.10+.

**Paper trading is the default and the fallback.** Live trading requires three
independent confirmations; miss any one of them and the run is demoted to
paper with the reason printed on screen. See [Safety rules](#safety-rules).

> **Not investment advice.** This is engineering scaffolding for studying a set
> of rules. Backtest numbers are not a forecast, and the defaults have not been
> tuned for profitability.

---

## Assets and strategies

| Asset | What it is | Timeframe | Strategy | Entry | Exit |
|---|---|---|---|---|---|
| **SPY** | S&P 500 ETF | 15 min | Mean reversion | Close ≤ 1.5σ below the 20-period SMA → long; ≥ 1.5σ above → short | Price returns to the SMA |
| **QQQ** | Nasdaq-100 ETF | 15 min | Mean reversion | Same, at **1.8σ** — QQQ is the more volatile of the two, so the band is widened rather than letting it trade on ordinary noise | Price returns to the SMA |
| **BTC/USD** | Bitcoin | 1 hour | Volume-confirmed breakout | Close above the prior 20-period high **and** volume ≥ 1.5× its 20-period average → long; same volume condition on a break of the 20-period low → short / close long | 2 × ATR trailing stop, or an opposite break |
| **GLD** | Gold **ETF proxy** | 4 hour | EMA trend following | 50 EMA crosses above 200 EMA → long | Cross back below → close / short; 3 × ATR trailing stop |
| **USO** | Crude oil **ETF proxy** | 4 hour | EMA trend following | Same | Same |

### GLD and USO are ETF proxies, not commodity futures

`GLD` and `USO` are exchange-traded funds, **not** futures contracts on gold or
crude oil. This matters for how you read any signal or result:

* **GLD** holds allocated gold bullion. It tracks spot gold closely, minus an
  ongoing expense ratio, so it drifts slightly below the metal over long
  horizons.
* **USO** holds **front-month crude oil futures and rolls them**. In contango
  (later-dated futures priced above nearer ones) each roll sells low and buys
  high, and the fund bleeds value against spot; in backwardation the roll works
  in its favour. USO's multi-year chart can therefore diverge sharply from the
  price of oil, and it has changed its holdings mix and executed a reverse split
  in the past.
* Both trade **only during US market hours**, so they gap over weekends and
  overnight news in a way an actual futures contract — which trades nearly
  around the clock — does not.

Treat every GLD/USO signal in this bot as a signal on the ETF itself. If you
want exposure to the underlying commodity, an ETF proxy is not the same
instrument, and Alpaca does not offer futures.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # optional: works without it, on synthetic data
python main.py backtest --months 6
python main.py paper --dry-run
```

Nothing here needs credentials to run. Without an Alpaca key the bot falls back
to a simulated broker that prints intended orders and sends nothing, and market
data falls back to a deterministic synthetic generator (loudly flagged as
`SYNTHETIC DATA` in every report that used it).

---

## Commands

```bash
python main.py backtest --months 6   # historical run, next-bar fills, costs included
python main.py paper --dry-run       # one cycle, print intended orders, send nothing
python main.py paper --once          # one cycle against the paper account
python main.py live --once           # one cycle against live — needs all three confirmations
python main.py status                # state, positions, drawdown, recent trades
python main.py stop --close-all      # cancel orders, flatten, persist STOPPED
python main.py resume                # clear STOPPED

python report.py morning --dry-run   # pre-session briefing (Telegram preview)
python report.py evening --dry-run   # end-of-day wrap (Telegram preview)
```

`report.py` is dry-run by default; pass `--send` to actually deliver to
Telegram.

---

## Safety rules

All twelve rules, where each one lives, and how it is verified.

| # | Rule | Implementation | Verified by |
|---|---|---|---|
| 1 | Paper is the default. With no environment at all, only `paper-api.alpaca.markets` is reachable | `settings.resolve_mode` | `test_live_requires_triple_confirm.py::test_empty_environment_is_paper` |
| 2 | Live requires **all three**: `ALPACA_PAPER=false`, `ALPACA_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY`, and an explicit live request on the command line. Any one missing → demote to paper and say why | `settings.resolve_mode` | the full 8-row truth table in `test_live_requires_triple_confirm.py` |
| 3 | Live start prints a red banner, current balance and worst-case loss per trade, then a 10-second countdown. Every log line carries the mode | `main.print_live_banner`, `settings.setup_logging` | manual; `mode=` appears on every line of `logs/bot.log` |
| 4 | Risk per trade = 1% of **current** equity. Quantity = `(equity × 0.01) ÷ stop distance`. Stop distance ≤ 0 → skip the order | `risk/manager.py::position_size` | `test_risk.py` (exactness, volatility invariance, skip cases) |
| 5 | 10% drawdown from peak equity → block new orders, cancel working orders, flatten positions, persist `STOPPED` | `risk/manager.py::RiskManager.trip_kill_switch` | `test_risk.py::test_kill_switch_stops_cancels_and_flattens` |
| 6 | `STOPPED` persists across restarts until a human runs `resume` | `portfolio.StateStore` + `state.json` | `test_risk.py::test_stopped_state_survives_a_restart` |
| 7 | Before every order: tradability, session hours, minimum quantity, duplicate orders, available cash | `risk/manager.py::pre_trade_checks` | `test_risk.py` (one test per check) |
| 8 | Equity ETFs trade the US session only; BTC/USD is 24/7 | `main.TradingEngine._process_symbol`, `pre_trade_checks` | `test_risk.py::test_crypto_trades_when_the_stock_market_is_closed` |
| 9 | Keys come from `.env` only and never reach logs, exception messages or reports | `settings.Secret`, `SecretFilter`, `install_exception_masking` | secrets render as `***REDACTED***` in every path |
| 10 | Backtest fills start on the bar **after** the signal bar — never the signal bar's close | `backtest.Backtester` (`fill_delay_bars ≥ 1`, fill at next bar's open) | `test_no_lookahead.py` |
| 11 | Commission and slippage assumptions live in `config.yaml` and are printed with every result | `backtest.CostModel`, `backtest.format_result`, `report.py` | `test_no_lookahead.py::test_costs_are_charged_and_reported` |
| 12 | Telegram is dry-run by default; a missing token gives a console preview, not an error | `report.send_telegram` | run `python report.py morning --dry-run` |

### The live-trading gate in detail

```
ALPACA_PAPER=false                              (environment)
ALPACA_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY     (environment, exact match)
`live` sub-command, or --live on `paper`        (command line)
```

Only all three together open the live endpoint. The `live` sub-command *is* the
explicit command-line request, so `python main.py live --once` satisfies the
third condition — and with the two environment variables absent it still
demotes to paper and tells you which ones were missing:

```
==========================================================================
  LIVE TRADING WAS REQUESTED BUT DEMOTED TO PAPER
    - ALPACA_PAPER is not 'false' (effective value: 'true', default is 'true')
    - ALPACA_LIVE_CONFIRM is not set
  Running against https://paper-api.alpaca.markets
==========================================================================
```

`resolve_mode` is a pure function with no I/O, and it is the only place in the
code base that can produce a live base URL. That is what makes the truth table
test exhaustive rather than indicative.

### Position sizing

The hard stop sits exactly one ATR(14) away from entry, and quantity is the
risk budget divided by that distance:

```
stop_distance = ATR(14)
quantity      = (equity × 1%) ÷ stop_distance
```

So a 1-ATR adverse move costs exactly 1% of equity on every instrument. A quiet
asset (small ATR) gets a large quantity, a violent one gets a small quantity,
and the dollars at risk never change. Two caps can only ever *reduce* the size:
`max_position_notional_pct` (35% of equity by default) and available cash.

### Correlation filter

While SPY **and** QQQ are both long, a new BTC/USD **long** is refused — the
book already holds the same risk-on bet twice. Shorts and exits are never
blocked.

---

## Files

```
main.py                          CLI, live banner, continuous loop
config.yaml                      assets, risk limits, cost assumptions, schedule
.env.example                     credential template (values stay empty here)
settings.py                      .env loading, secret masking, resolve_mode  ← safety core
data.py                          bars: cache → Alpaca API → synthetic fallback
indicators.py                    SMA / EMA / stddev / ATR / rolling high-low
broker.py                        BrokerBase, AlpacaBroker, DryRunBroker, retry+backoff
portfolio.py                     positions, state.json, trades.csv, daily_pnl.csv
backtest.py                      next-bar-fill engine, cost model, metrics
report.py                        morning / evening reports, Telegram dry-run
strategies/
  base.py                        Signal + Strategy interface
  mean_reversion.py              SPY (1.5σ), QQQ (1.8σ)
  breakout.py                    BTC/USD, volume-confirmed, 2×ATR trail
  trend_following.py             GLD / USO, 50-200 EMA, 3×ATR trail
risk/manager.py                  sizing, hard stops, kill switch, pre-trade checks
tests/                           risk, triple-confirmation, no-lookahead
```

Runtime artefacts (all git-ignored): `trades.csv`, `daily_pnl.csv`,
`state.json`, `logs/bot.log`, `data/cache/`.

### Output files

`trades.csv` — one row per closed trade:

```
timestamp,symbol,side,entry_price,exit_price,pnl,qty,
strategy,entry_time,fees,slippage,return_pct,exit_reason,mode
```

`daily_pnl.csv` — one row per day (repeated runs update the day's row in place):

```
date,starting_equity,ending_equity,realized_pnl,unrealized_pnl,
trades,max_drawdown_pct,mode
```

---

## Market data

Every request resolves in this order:

1. **CSV cache** under `data/cache/` — fast, offline, reproducible;
2. **Alpaca historical API** — needs credentials in `.env`;
3. **Synthetic generator** — deterministic geometric Brownian motion, seeded
   per symbol.

Step 3 exists so the whole pipeline can be exercised with no credentials and no
network. Anything that consumed synthetic bars is stamped:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  SYNTHETIC DATA — no Alpaca credentials and no cache were available.
  These bars were generated locally. The numbers below are a
  pipeline demonstration, NOT a claim about real performance.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Synthetic equity bars follow US trading hours: minute bars cover the regular
session, hourly-and-larger buckets span the extended session (04:00–20:00 ET),
matching how Alpaca aggregates. Without that distinction a 4-hour series yields
barely one bar a day and the 200-EMA would need years of history to warm up.

---

## Backtesting

The engine is event-driven across all five assets on one shared equity curve,
so the correlation filter and the drawdown kill switch behave exactly as they
do live. Signals are evaluated on bar `i` and executed at the **open of bar
`i+1`**, with slippage applied against the order.

The no-look-ahead guarantee is structural, not a convention:

1. `Strategy.evaluate` receives `bars.iloc[:i+1]` — a future bar is not
   reachable from inside a strategy;
2. an actionable signal schedules a pending order at index `i+1`, and every
   fill records both indices;
3. the engine asserts `fill_index == signal_index + 1` before returning a
   result, and `fill_delay_bars = 0` is rejected at construction time.

`tests/test_no_lookahead.py` also rewrites the tail of a bar series and checks
that no earlier signal or fill moves.

The drawdown kill switch applies during backtests too: if the equity curve
falls 10% from its peak, trading halts for the remainder of the period and the
summary says so. That is the live system's real behaviour, so the backtest
shows it rather than hiding it.

---

## Continuous operation

`python main.py paper` runs a loop that wakes every 30 seconds and checks each
strategy on its own cadence (15 min / 1 hour / 4 hour, configurable under
`schedule.intervals`). Each cycle:

1. read the account (retried with exponential backoff on API failure);
2. update peak equity and evaluate the drawdown guard;
3. bail out immediately if the state is `STOPPED`;
4. skip equity ETFs when the US market is closed; BTC/USD keeps running;
5. per symbol: ratchet the trailing stop, evaluate the signal, size, run the
   pre-trade checks, submit.

API disconnects are retried with jittered exponential backoff and then logged;
one failing symbol never takes down the loop.

---

## Tests

```bash
pytest -q
```

124 tests, no network and no credentials required. `tests/conftest.py` scrubs
`ALPACA_*` from the environment before every test, so a developer's real `.env`
can never turn a test run into a live-trading attempt, and the broker double
raises if anything tries to submit an order.

---

## Configuration

Everything tunable is in `config.yaml`: per-asset strategy parameters and
timeframes, `risk.per_trade_pct`, `risk.max_drawdown_pct`, the ATR period, the
cost assumptions, the schedule and the file paths. Nothing secret belongs
there — credentials come from `.env` only.
