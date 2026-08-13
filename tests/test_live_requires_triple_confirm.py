"""Safety rule 2 — the live-trading gate.

Live trading requires all three of:

  1. ``ALPACA_PAPER=false``
  2. ``ALPACA_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY``
  3. an explicit live request on the command line

These tests walk every one of the eight combinations.  Exactly one of them —
all three present — may open the live path; the other seven must resolve to
the paper endpoint, and the three that asked for live must say why they were
demoted.
"""

from __future__ import annotations

import itertools

import pytest

from broker import AlpacaBroker, DryRunBroker, build_broker
from settings import (
    LIVE_CONFIRM_TOKEN,
    LIVE_ENDPOINT,
    PAPER_ENDPOINT,
    Credentials,
    Secret,
    resolve_mode,
)


def make_env(paper_false: bool, confirm: bool) -> dict[str, str]:
    env: dict[str, str] = {}
    env["ALPACA_PAPER"] = "false" if paper_false else "true"
    if confirm:
        env["ALPACA_LIVE_CONFIRM"] = LIVE_CONFIRM_TOKEN
    return env


ALL_COMBINATIONS = list(itertools.product([False, True], repeat=3))


# --------------------------------------------------------------------------
# The full truth table
# --------------------------------------------------------------------------


@pytest.mark.parametrize("paper_false,confirm,cli_live", ALL_COMBINATIONS)
def test_only_all_three_confirmations_open_live(paper_false, confirm, cli_live):
    decision = resolve_mode(make_env(paper_false, confirm), cli_live=cli_live)
    expected_live = paper_false and confirm and cli_live

    assert decision.live is expected_live
    assert decision.paper is not expected_live
    if expected_live:
        assert decision.endpoint == LIVE_ENDPOINT
        assert decision.label == "LIVE"
        assert decision.reasons == ()
    else:
        # Rule 1/2: anything short of three confirmations lands on paper.
        assert decision.endpoint == PAPER_ENDPOINT
        assert decision.label == "PAPER"


@pytest.mark.parametrize("paper_false,confirm,cli_live", ALL_COMBINATIONS)
def test_exactly_one_combination_is_live(paper_false, confirm, cli_live):
    """Sanity check on the table itself: only 1 of 8 rows may be live."""
    live_rows = [
        c for c in ALL_COMBINATIONS if resolve_mode(make_env(c[0], c[1]), cli_live=c[2]).live
    ]
    assert live_rows == [(True, True, True)]


# --------------------------------------------------------------------------
# Demotion: each missing condition is reported
# --------------------------------------------------------------------------


def test_missing_paper_flag_demotes_and_explains():
    decision = resolve_mode(make_env(paper_false=False, confirm=True), cli_live=True)
    assert decision.live is False
    assert decision.demoted is True
    assert any("ALPACA_PAPER" in r for r in decision.reasons)


def test_missing_confirm_token_demotes_and_explains():
    decision = resolve_mode(make_env(paper_false=True, confirm=False), cli_live=True)
    assert decision.live is False
    assert decision.demoted is True
    assert any("ALPACA_LIVE_CONFIRM" in r for r in decision.reasons)


def test_missing_cli_flag_demotes_and_explains():
    decision = resolve_mode(make_env(paper_false=True, confirm=True), cli_live=False)
    assert decision.live is False
    # Nobody asked for live, so this is a plain paper run, not a demotion.
    assert decision.demoted is False
    assert any("--live" in r for r in decision.reasons)


def test_every_missing_condition_gets_its_own_reason():
    # Nothing supplied at all: all three conditions are reported as missing.
    bare = resolve_mode({}, cli_live=False)
    assert bare.live is False
    assert len(bare.reasons) == 3

    # Asking for live on the command line satisfies one of the three, so only
    # the two environment conditions remain outstanding.
    asked = resolve_mode({}, cli_live=True)
    assert asked.live is False
    assert asked.demoted is True
    assert len(asked.reasons) == 2
    assert not any("--live" in r for r in asked.reasons)


def test_demotion_message_is_human_readable():
    decision = resolve_mode(make_env(paper_false=True, confirm=False), cli_live=True)
    text = decision.describe()
    assert "DEMOTED TO PAPER" in text
    assert "ALPACA_LIVE_CONFIRM" in text


# --------------------------------------------------------------------------
# Rule 1: no environment at all => paper
# --------------------------------------------------------------------------


def test_empty_environment_is_paper():
    decision = resolve_mode({}, cli_live=False)
    assert decision.live is False
    assert decision.endpoint == PAPER_ENDPOINT


def test_default_when_alpaca_paper_unset_is_paper():
    decision = resolve_mode({"ALPACA_LIVE_CONFIRM": LIVE_CONFIRM_TOKEN}, cli_live=True)
    assert decision.live is False
    assert any("ALPACA_PAPER" in r for r in decision.reasons)


# --------------------------------------------------------------------------
# The confirmation token must match exactly
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    [
        "",
        "i_understand_real_money",
        "I_UNDERSTAND_REAL_MONEY ",
        " I_UNDERSTAND_REAL_MONEY",
        "I_UNDERSTAND_REAL_MONEYY",
        "YES",
        "true",
    ],
)
def test_near_miss_tokens_do_not_unlock_live(token):
    env = {"ALPACA_PAPER": "false", "ALPACA_LIVE_CONFIRM": token}
    decision = resolve_mode(env, cli_live=True)
    assert decision.live is False
    assert decision.endpoint == PAPER_ENDPOINT


@pytest.mark.parametrize("value", ["FALSE", "False", "false", " false "])
def test_alpaca_paper_false_is_case_and_space_insensitive(value):
    env = {"ALPACA_PAPER": value, "ALPACA_LIVE_CONFIRM": LIVE_CONFIRM_TOKEN}
    assert resolve_mode(env, cli_live=True).live is True


@pytest.mark.parametrize("value", ["true", "0", "no", "", "yes", "1"])
def test_anything_other_than_false_keeps_paper(value):
    env = {"ALPACA_PAPER": value, "ALPACA_LIVE_CONFIRM": LIVE_CONFIRM_TOKEN}
    assert resolve_mode(env, cli_live=True).live is False


# --------------------------------------------------------------------------
# The decision is what the broker consumes
# --------------------------------------------------------------------------


def fake_credentials() -> Credentials:
    return Credentials(
        api_key=Secret("PKTESTKEY0000000"),
        secret_key=Secret("secrettestvalue000000"),
        telegram_token=Secret(""),
        telegram_chat_id=Secret(""),
    )


def test_live_decision_builds_a_live_broker(config):
    decision = resolve_mode(make_env(True, True), cli_live=True)
    broker = build_broker(decision, fake_credentials(), config)
    assert isinstance(broker, AlpacaBroker)
    # The SDK client is lazy, so nothing has been contacted; what matters is
    # that the flag the client would receive is the live one.
    assert broker.decision.live is True
    assert broker.decision.paper is False
    assert broker.decision.endpoint == LIVE_ENDPOINT
    assert broker._client is None


@pytest.mark.parametrize(
    "paper_false,confirm,cli_live",
    [c for c in ALL_COMBINATIONS if not all(c)],
)
def test_incomplete_confirmation_builds_a_paper_broker(paper_false, confirm, cli_live, config):
    decision = resolve_mode(make_env(paper_false, confirm), cli_live=cli_live)
    broker = build_broker(decision, fake_credentials(), config)
    assert isinstance(broker, AlpacaBroker)
    assert broker.decision.paper is True
    assert broker.decision.endpoint == PAPER_ENDPOINT


def test_without_credentials_the_broker_cannot_trade_at_all(config):
    decision = resolve_mode(make_env(True, True), cli_live=True)
    empty = Credentials(Secret(""), Secret(""), Secret(""), Secret(""))
    broker = build_broker(decision, empty, config)
    assert isinstance(broker, DryRunBroker)
    result = broker.submit_order("SPY", "LONG", 10, stop_price=1.0)
    assert result.submitted is False
    assert broker.submitted == [result]


# --------------------------------------------------------------------------
# End to end through the CLI
# --------------------------------------------------------------------------


def test_cli_live_once_without_env_runs_on_paper(workdir, monkeypatch, capsys):
    """`python main.py live --once` with no env must not reach a live account."""
    import main

    captured: dict[str, object] = {}
    real_build = main.build_runtime

    def spy(args, cli_live, force_dry_run=False):
        rt = real_build(args, cli_live, force_dry_run)
        captured["decision"] = rt.decision
        captured["broker"] = rt.broker
        return rt

    monkeypatch.setattr(main, "build_runtime", spy)
    # If the live banner were ever reached, this would make the test fail loudly.
    monkeypatch.setattr(
        main,
        "print_live_banner",
        lambda *a, **k: pytest.fail("live banner reached without confirmations"),
    )

    exit_code = main.main(["live", "--once"])

    assert exit_code == 0
    decision = captured["decision"]
    assert decision.live is False
    assert decision.demoted is True
    assert decision.endpoint == PAPER_ENDPOINT
    # No credentials in the environment => simulated broker, nothing sendable.
    broker = captured["broker"]
    assert isinstance(broker, DryRunBroker)
    assert all(o.submitted is False for o in broker.submitted)

    out = capsys.readouterr().out
    assert "DEMOTED TO PAPER" in out


def test_cli_paper_dry_run_never_submits(workdir, monkeypatch):
    import main

    captured: dict[str, object] = {}
    real_build = main.build_runtime

    def spy(args, cli_live, force_dry_run=False):
        rt = real_build(args, cli_live, force_dry_run)
        captured["broker"] = rt.broker
        captured["decision"] = rt.decision
        return rt

    monkeypatch.setattr(main, "build_runtime", spy)
    assert main.main(["paper", "--dry-run"]) == 0

    broker = captured["broker"]
    assert isinstance(broker, DryRunBroker)
    assert broker.label == "DRY-RUN"
    assert all(o.submitted is False for o in broker.submitted)
    assert captured["decision"].paper is True


def test_paper_subcommand_with_live_flag_still_needs_the_env(workdir, monkeypatch):
    """`paper --live` counts as the CLI confirmation but not as the other two."""
    import main

    parser = main.build_parser()
    args = parser.parse_args(["paper", "--live", "--once"])
    assert args.live is True

    decision = resolve_mode({}, cli_live=bool(args.live))
    assert decision.live is False
    assert decision.demoted is True
