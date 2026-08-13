"""Configuration, secret handling and paper/live mode resolution.

This module is the single place where the paper-vs-live decision is made and
where API credentials are read. Nothing else may build a live endpoint URL or
touch ``os.environ`` for credentials.

Kiwoom issues **separate app keys for the mock and live environments**, which
lets this bot go one step further than a shared-key broker: when the gate
resolves to paper, :func:`load_credentials` does not read the live key at all.
A mis-assembled URL therefore still cannot reach a live account — the
credential simply is not in memory.

Safety rules implemented here:
  * Rule 1  — with no environment at all we always resolve to the mock endpoint.
  * Rule 2  — live trading requires *three* independent confirmations.
  * Rule 3  — the resolved mode is attached to every log line.
  * Rule 9  — credentials come from ``.env`` only and are masked everywhere.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Kiwoom REST API hosts. Verify against the official developer portal
# (https://openapi.kiwoom.com/) before trading real money.
PAPER_ENDPOINT = "https://mockapi.kiwoom.com"
LIVE_ENDPOINT = "https://api.kiwoom.com"

#: The exact, case-sensitive value ``KIWOOM_LIVE_CONFIRM`` must carry.
LIVE_CONFIRM_TOKEN = "I_UNDERSTAND_REAL_MONEY"

#: Environment variables holding secrets. Never printed.
SECRET_ENV_KEYS = (
    "KIWOOM_PAPER_APP_KEY",
    "KIWOOM_PAPER_SECRET_KEY",
    "KIWOOM_LIVE_APP_KEY",
    "KIWOOM_LIVE_SECRET_KEY",
    "KIWOOM_ACCOUNT_NO",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
)

REDACTED = "***REDACTED***"

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Secret handling (rule 9)
# --------------------------------------------------------------------------


class Secret:
    """A string that refuses to render itself.

    ``repr``/``str``/``format`` all yield ``***REDACTED***`` so a secret cannot
    leak through an f-string, a log call or a traceback. The real value is
    reachable only through :meth:`reveal`, called in exactly one place (the
    Kiwoom token request in ``broker.py``).
    """

    __slots__ = ("_value",)

    def __init__(self, value: str | None) -> None:
        self._value = value or ""

    def reveal(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __len__(self) -> int:
        return len(REDACTED)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Secret):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("Secret", self._value))

    def __str__(self) -> str:
        return REDACTED

    __repr__ = __str__

    def __format__(self, spec: str) -> str:
        return REDACTED


def _secret_values(env: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    values: list[str] = []
    for key in SECRET_ENV_KEYS:
        raw = env.get(key)
        # Very short values would cause absurd over-masking.
        if raw and len(raw) >= 8:
            values.append(raw)
    return values


def mask_text(text: str, env: Mapping[str, str] | None = None) -> str:
    """Replace every known secret occurring in *text* with ``***REDACTED***``."""
    if not text:
        return text
    for value in _secret_values(env):
        if value in text:
            text = text.replace(value, REDACTED)
    return text


class SecretFilter(logging.Filter):
    """Logging filter that scrubs credentials from messages and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not _secret_values():
            return True
        if isinstance(record.msg, str):
            record.msg = mask_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: mask_text(v) if isinstance(v, str) else v for k, v in record.args.items()
                }
            else:
                record.args = tuple(
                    mask_text(a) if isinstance(a, str) else a for a in record.args
                )
        return True


def install_exception_masking() -> None:
    """Ensure uncaught tracebacks cannot print a credential."""
    import traceback

    def _hook(exc_type, exc_value, exc_tb):  # pragma: no cover - process exit path
        rendered = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        sys.stderr.write(mask_text(rendered))

    sys.excepthook = _hook


# --------------------------------------------------------------------------
# Mode resolution (rules 1 & 2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeDecision:
    """Outcome of the three-way live-trading gate."""

    live: bool
    demoted: bool
    endpoint: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    checks: Mapping[str, bool] = field(default_factory=dict)

    @property
    def paper(self) -> bool:
        return not self.live

    @property
    def label(self) -> str:
        return "LIVE" if self.live else "PAPER"

    def describe(self) -> str:
        if self.live:
            return "LIVE trading enabled (all three confirmations present)."
        if self.demoted:
            return "Live trading was requested but DEMOTED TO PAPER: " + "; ".join(self.reasons)
        return "PAPER trading (default)."


def resolve_mode(
    env: Mapping[str, str] | None = None,
    cli_live: bool = False,
) -> ModeDecision:
    """Decide between mock and live trading.

    Live trading requires **all three** of:

    1. ``KIWOOM_PAPER=false`` in the environment,
    2. ``KIWOOM_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY`` in the environment,
    3. an explicit live request on the command line (``live`` sub-command or
       ``--live``), passed here as ``cli_live=True``.

    Any missing condition demotes the run to paper and records a human readable
    reason. This function is pure — no I/O, no globals, no logging — which is
    what makes the triple-confirmation test exhaustive and cheap.
    """
    env = os.environ if env is None else env

    paper_raw = str(env.get("KIWOOM_PAPER", "true")).strip().lower()
    check_paper_false = paper_raw == "false"

    confirm_raw = env.get("KIWOOM_LIVE_CONFIRM")
    check_confirm = confirm_raw == LIVE_CONFIRM_TOKEN

    check_cli = bool(cli_live)

    reasons: list[str] = []
    if not check_paper_false:
        reasons.append(
            f"KIWOOM_PAPER is not 'false' (effective value: {paper_raw!r}, default is 'true')"
        )
    if not check_confirm:
        if confirm_raw is None:
            reasons.append("KIWOOM_LIVE_CONFIRM is not set")
        else:
            reasons.append("KIWOOM_LIVE_CONFIRM does not exactly match the required token")
    if not check_cli:
        reasons.append(
            "no explicit live request on the command line "
            "(use the 'live' sub-command or --live)"
        )

    live = check_paper_false and check_confirm and check_cli
    return ModeDecision(
        live=live,
        demoted=(not live) and check_cli,
        endpoint=LIVE_ENDPOINT if live else PAPER_ENDPOINT,
        reasons=tuple(reasons),
        checks={
            "KIWOOM_PAPER=false": check_paper_false,
            "KIWOOM_LIVE_CONFIRM": check_confirm,
            "--live / live sub-command": check_cli,
        },
    )


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Credentials:
    app_key: Secret
    secret_key: Secret
    account_no: Secret
    telegram_token: Secret
    telegram_chat_id: Secret
    #: Which key set was loaded. Should always match the ModeDecision.
    loaded_for: str = "PAPER"

    @property
    def has_kiwoom(self) -> bool:
        return bool(self.app_key) and bool(self.secret_key)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_token) and bool(self.telegram_chat_id)


def load_env(dotenv_path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Load ``.env`` into ``os.environ`` and return a snapshot.

    Credentials are read from ``.env`` only (rule 9). A missing ``.env`` is not
    an error: with no environment at all we simply stay on paper.
    """
    path = Path(dotenv_path) if dotenv_path else PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        if path.exists():
            load_dotenv(path, override=False)
    return dict(os.environ)


def load_credentials(
    env: Mapping[str, str] | None = None,
    decision: ModeDecision | None = None,
) -> Credentials:
    """Load the key set matching *decision* — and only that key set.

    This is the extra safety layer Kiwoom's split keys make possible: on a
    paper run the live app key is never read out of the environment, so no
    amount of downstream misconfiguration can authenticate against a real
    account.
    """
    env = os.environ if env is None else env
    live = bool(decision and decision.live)
    prefix = "KIWOOM_LIVE_" if live else "KIWOOM_PAPER_"

    return Credentials(
        app_key=Secret(env.get(f"{prefix}APP_KEY")),
        secret_key=Secret(env.get(f"{prefix}SECRET_KEY")),
        account_no=Secret(env.get("KIWOOM_ACCOUNT_NO")),
        telegram_token=Secret(env.get("TELEGRAM_BOT_TOKEN")),
        telegram_chat_id=Secret(env.get("TELEGRAM_CHAT_ID")),
        loaded_for="LIVE" if live else "PAPER",
    )


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read ``config.yaml``."""
    import yaml

    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------
# Logging (rule 3: every line carries the mode)
# --------------------------------------------------------------------------


class _ModeAdapter(logging.Filter):
    def __init__(self, mode_label: str) -> None:
        super().__init__()
        self.mode_label = mode_label

    def filter(self, record: logging.LogRecord) -> bool:
        record.mode = self.mode_label
        return True


def setup_logging(
    decision: ModeDecision,
    log_file: str | os.PathLike[str] = "logs/bot.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure logging with mode tagging and secret scrubbing."""
    logger = logging.getLogger("bot")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | mode=%(mode)s | %(name)s | %(message)s"
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    try:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("log file unavailable, continuing with console logging only")

    logger.addFilter(_ModeAdapter(decision.label))
    logger.addFilter(SecretFilter())
    for handler in logger.handlers:
        handler.addFilter(_ModeAdapter(decision.label))
        handler.addFilter(SecretFilter())

    install_exception_masking()
    return logger


def redact_mapping(data: Mapping[str, Any], keys: Iterable[str] = SECRET_ENV_KEYS) -> dict[str, Any]:
    """Return a copy of *data* with any secret-looking key redacted."""
    lowered = {k.lower() for k in keys}
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in lowered or any(
            token in key.lower() for token in ("secret", "token", "app_key", "password", "account")
        ):
            out[key] = REDACTED
        else:
            out[key] = value
    return out
