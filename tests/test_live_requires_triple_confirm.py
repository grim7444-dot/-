"""Safety rule 2 — the live-trading gate.

Live trading requires all three of:

  1. ``KIWOOM_PAPER=false``
  2. ``KIWOOM_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY``
  3. an explicit live request on the command line

These tests walk every one of the eight combinations. Exactly one of them —
all three present — may open the live path; the other seven must resolve to
the mock endpoint, and the three that asked for live must say why they were
demoted.

Kiwoom issues separate mock and live app keys, which supports one extra
guarantee tested here: on a paper run the **live key is never read out of the
environment**, so no downstream misconfiguration can authenticate against a
real account.
"""

from __future__ import annotations

import itertools

import pytest

from broker import DryRunBroker, KiwoomBroker, build_broker
from settings import (
    LIVE_CONFIRM_TOKEN,
    LIVE_ENDPOINT,
    PAPER_ENDPOINT,
    Credentials,
    Secret,
    load_credentials,
    resolve_mode,
)

PAPER_KEY = "PAPERAPPKEY000000"
PAPER_SECRET = "papersecretvalue000000"
LIVE_KEY = "LIVEAPPKEY0000000"
LIVE_SECRET = "livesecretvalue0000000"


def make_env(paper_false: bool, confirm: bool, with_keys: bool = False) -> dict[str, str]:
    env: dict[str, str] = {"KIWOOM_PAPER": "false" if paper_false else "true"}
    if confirm:
        env["KIWOOM_LIVE_CONFIRM"] = LIVE_CONFIRM_TOKEN
    if with_keys:
        env.update(
            {
                "KIWOOM_PAPER_APP_KEY": PAPER_KEY,
                "KIWOOM_PAPER_SECRET_KEY": PAPER_SECRET,
                "KIWOOM_LIVE_APP_KEY": LIVE_KEY,
                "KIWOOM_LIVE_SECRET_KEY": LIVE_SECRET,
            }
        )
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
        # Rules 1/2: anything short of three confirmations lands on paper.
        assert decision.endpoint == PAPER_ENDPOINT
        assert decision.label == "PAPER"


def test_exactly_one_combination_is_live():
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
    assert any("KIWOOM_PAPER" in r for r in decision.reasons)


def test_missing_confirm_token_demotes_and_explains():
    decision = resolve_mode(make_env(paper_false=True, confirm=False), cli_live=True)
    assert decision.live is False
    assert decision.demoted is True
    assert any("KIWOOM_LIVE_CONFIRM" in r for r in decision.reasons)


def test_missing_cli_flag_demotes_and_explains():
    decision = resolve_mode(make_env(paper_false=True, confirm=True), cli_live=False)
    assert decision.live is False
    # Nobody asked for live, so this is a plain paper run, not a demotion.
    assert decision.demoted is False
    assert any("--live" in r for r in decision.reasons)


def test_every_missing_condition_gets_its_own_reason():
    bare = resolve_mode({}, cli_live=False)
    assert bare.live is False
    assert len(bare.reasons) == 3

    asked = resolve_mode({}, cli_live=True)
    assert asked.live is False
    assert asked.demoted is True
    assert len(asked.reasons) == 2
    assert not any("--live" in r for r in asked.reasons)


def test_demotion_message_is_human_readable():
    decision = resolve_mode(make_env(paper_false=True, confirm=False), cli_live=True)
    text = decision.describe()
    assert "DEMOTED TO PAPER" in text
    assert "KIWOOM_LIVE_CONFIRM" in text


# --------------------------------------------------------------------------
# Rule 1: no environment at all => paper
# --------------------------------------------------------------------------


def test_empty_environment_is_paper():
    decision = resolve_mode({}, cli_live=False)
    assert decision.live is False
    assert decision.endpoint == PAPER_ENDPOINT


def test_default_when_kiwoom_paper_unset_is_paper():
    decision = resolve_mode({"KIWOOM_LIVE_CONFIRM": LIVE_CONFIRM_TOKEN}, cli_live=True)
    assert decision.live is False
    assert any("KIWOOM_PAPER" in r for r in decision.reasons)


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
    env = {"KIWOOM_PAPER": "false", "KIWOOM_LIVE_CONFIRM": token}
    decision = resolve_mode(env, cli_live=True)
    assert decision.live is False
    assert decision.endpoint == PAPER_ENDPOINT


@pytest.mark.parametrize("value", ["FALSE", "False", "false", " false "])
def test_kiwoom_paper_false_is_case_and_space_insensitive(value):
    env = {"KIWOOM_PAPER": value, "KIWOOM_LIVE_CONFIRM": LIVE_CONFIRM_TOKEN}
    assert resolve_mode(env, cli_live=True).live is True


@pytest.mark.parametrize("value", ["true", "0", "no", "", "yes", "1"])
def test_anything_other_than_false_keeps_paper(value):
    env = {"KIWOOM_PAPER": value, "KIWOOM_LIVE_CONFIRM": LIVE_CONFIRM_TOKEN}
    assert resolve_mode(env, cli_live=True).live is False


# --------------------------------------------------------------------------
# The live key is not even loaded on a paper run
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "paper_false,confirm,cli_live",
    [c for c in ALL_COMBINATIONS if not all(c)],
)
def test_paper_runs_never_load_the_live_key(paper_false, confirm, cli_live):
    env = make_env(paper_false, confirm, with_keys=True)
    decision = resolve_mode(env, cli_live=cli_live)
    assert decision.paper is True

    credentials = load_credentials(env, decision)
    assert credentials.loaded_for == "PAPER"
    assert credentials.app_key.reveal() == PAPER_KEY
    # The decisive assertion: the live secret is simply not in memory.
    assert credentials.app_key.reveal() != LIVE_KEY
    assert credentials.secret_key.reveal() != LIVE_SECRET


def test_live_run_loads_the_live_key():
    env = make_env(True, True, with_keys=True)
    decision = resolve_mode(env, cli_live=True)
    assert decision.live is True

    credentials = load_credentials(env, decision)
    assert credentials.loaded_for == "LIVE"
    assert credentials.app_key.reveal() == LIVE_KEY
    assert credentials.secret_key.reveal() == LIVE_SECRET


def test_credentials_never_render_their_value():
    env = make_env(True, True, with_keys=True)
    credentials = load_credentials(env, resolve_mode(env, cli_live=True))
    assert LIVE_KEY not in f"{credentials.app_key}"
    assert LIVE_KEY not in repr(credentials.app_key)
    assert LIVE_SECRET not in str(credentials.secret_key)


# --------------------------------------------------------------------------
# The decision is what the broker consumes
# --------------------------------------------------------------------------


def test_live_decision_builds_a_live_broker(config):
    env = make_env(True, True, with_keys=True)
    decision = resolve_mode(env, cli_live=True)
    broker = build_broker(decision, load_credentials(env, decision), config)

    assert isinstance(broker, KiwoomBroker)
    assert broker.decision.live is True
    assert broker.base_url == LIVE_ENDPOINT
    # No token has been requested, so nothing has been contacted.
    assert broker._token == ""


@pytest.mark.parametrize(
    "paper_false,confirm,cli_live",
    [c for c in ALL_COMBINATIONS if not all(c)],
)
def test_incomplete_confirmation_builds_a_paper_broker(paper_false, confirm, cli_live, config):
    env = make_env(paper_false, confirm, with_keys=True)
    decision = resolve_mode(env, cli_live=cli_live)
    broker = build_broker(decision, load_credentials(env, decision), config)

    assert isinstance(broker, KiwoomBroker)
    assert broker.decision.paper is True
    assert broker.base_url == PAPER_ENDPOINT


def test_broker_refuses_mismatched_credentials(config):
    """A live decision paired with paper credentials must not trade."""
    from broker import BrokerError

    decision = resolve_mode(make_env(True, True), cli_live=True)
    paper_credentials = Credentials(
        app_key=Secret(PAPER_KEY),
        secret_key=Secret(PAPER_SECRET),
        account_no=Secret(""),
        telegram_token=Secret(""),
        telegram_chat_id=Secret(""),
        loaded_for="PAPER",
    )
    with pytest.raises(BrokerError, match="does not match"):
        build_broker(decision, paper_credentials, config)


def test_without_credentials_the_broker_cannot_trade_at_all(config):
    env = make_env(True, True)  # confirmations present, but no keys
    decision = resolve_mode(env, cli_live=True)
    broker = build_broker(decision, load_credentials(env, decision), config)

    assert isinstance(broker, DryRunBroker)
    result = broker.submit_order("005930", "LONG", 10, stop_price=1000)
    assert result.submitted is False


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
        captured["credentials"] = rt.credentials
        return rt

    monkeypatch.setattr(main, "build_runtime", spy)
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
    assert captured["credentials"].loaded_for == "PAPER"

    broker = captured["broker"]
    assert isinstance(broker, DryRunBroker)
    assert all(o.submitted is False for o in broker.submitted)

    assert "DEMOTED TO PAPER" in capsys.readouterr().out


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


def test_check_command_never_submits_an_order(workdir, monkeypatch):
    """`check` is the read-only probe used before pointing at a live account."""
    import main

    captured: dict[str, object] = {}
    real_build = main.build_runtime

    def spy(args, cli_live, force_dry_run=False):
        rt = real_build(args, cli_live, force_dry_run)
        captured["broker"] = rt.broker
        captured["decision"] = rt.decision
        return rt

    monkeypatch.setattr(main, "build_runtime", spy)
    # No credentials, so the probe reports it cannot connect and exits non-zero.
    assert main.main(["check"]) == 1

    broker = captured["broker"]
    assert isinstance(broker, DryRunBroker)
    assert broker.submitted == []
    assert broker.cancelled == 0
    assert broker.closed == 0


def test_check_with_live_flag_still_needs_the_env(workdir, monkeypatch):
    """`check --live` obeys the same gate as trading: no env, no live account."""
    import main

    captured: dict[str, object] = {}
    real_build = main.build_runtime

    def spy(args, cli_live, force_dry_run=False):
        rt = real_build(args, cli_live, force_dry_run)
        captured["decision"] = rt.decision
        captured["credentials"] = rt.credentials
        return rt

    monkeypatch.setattr(main, "build_runtime", spy)
    main.main(["check", "--live"])

    assert captured["decision"].live is False
    assert captured["decision"].endpoint == PAPER_ENDPOINT
    assert captured["credentials"].loaded_for == "PAPER"


def test_paper_subcommand_with_live_flag_still_needs_the_env(workdir):
    """`paper --live` counts as the CLI confirmation but not as the other two."""
    import main

    args = main.build_parser().parse_args(["paper", "--live", "--once"])
    assert args.live is True

    decision = resolve_mode({}, cli_live=bool(args.live))
    assert decision.live is False
    assert decision.demoted is True


# --------------------------------------------------------------------------
# `setup` writes credentials without leaking them
# --------------------------------------------------------------------------


def test_setup_writes_values_and_never_prints_them(workdir, monkeypatch, capsys):
    import main

    (workdir / ".env").write_text(
        "KIWOOM_PAPER_APP_KEY=\nKIWOOM_PAPER_SECRET_KEY=\nKIWOOM_PAPER=true\n",
        encoding="utf-8",
    )
    answers = iter([PAPER_KEY, PAPER_SECRET, "12345678901"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))

    assert main.main(["setup"]) == 0

    written = (workdir / ".env").read_text(encoding="utf-8")
    assert f"KIWOOM_PAPER_APP_KEY={PAPER_KEY}" in written
    assert f"KIWOOM_PAPER_SECRET_KEY={PAPER_SECRET}" in written
    assert "KIWOOM_ACCOUNT_NO=12345678901" in written
    # Existing unrelated lines survive, and nothing is duplicated.
    assert "KIWOOM_PAPER=true" in written
    assert written.count("KIWOOM_PAPER_APP_KEY=") == 1

    # Safety rule 9: the console shows lengths, never the secrets themselves.
    out = capsys.readouterr().out
    assert PAPER_KEY not in out
    assert PAPER_SECRET not in out
    assert "43 chars" in out or "chars" in out


def test_setup_live_does_not_flip_the_gate_itself(workdir, monkeypatch):
    """Writing the LIVE keys must not also grant the two env confirmations."""
    import main

    (workdir / ".env").write_text("KIWOOM_PAPER=true\n", encoding="utf-8")
    answers = iter([LIVE_KEY, LIVE_SECRET, "12345678901"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))

    assert main.main(["setup", "--live"]) == 0

    written = (workdir / ".env").read_text(encoding="utf-8")
    assert f"KIWOOM_LIVE_APP_KEY={LIVE_KEY}" in written
    # The human half of the gate is untouched.
    assert "KIWOOM_PAPER=true" in written
    assert "KIWOOM_PAPER=false" not in written
    assert "KIWOOM_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY" not in written
