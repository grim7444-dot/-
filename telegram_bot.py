"""Telegram alerts and command handler.

Alerts  — one-way push on entry / exit / stop-loss / kill-switch events.
Commands — long-poll in a daemon thread; only messages from the configured
           chat_id are accepted to prevent anyone with the bot link from
           controlling the account.

Safety:
  • The bot token is revealed only inside HTTP request payloads and never
    logged.  mask_text() handles accidental leakage in exception messages.
  • Commands that touch orders (/close_all) log a WARNING so the audit trail
    is clear.
  • dry_run=True (default) prints previews and starts no polling thread —
    identical behaviour to the report.py convention.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from settings import mask_text

logger = logging.getLogger("bot.telegram")

_MAX_LEN = 4000  # Telegram hard limit is 4096; stay clear


class TelegramNotifier:
    """Push one-way alerts to a Telegram chat."""

    def __init__(self, token: str, chat_id: str, dry_run: bool = True) -> None:
        self._token = token
        self._chat_id = chat_id
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------

    def send(self, text: str, parse_mode: str = "Markdown") -> None:
        if self.dry_run or not self._token or not self._chat_id:
            logger.debug("[TG preview] %s", text[:120])
            return
        try:
            import requests
            requests.post(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": text[:_MAX_LEN],
                    "parse_mode": parse_mode,
                },
                timeout=10.0,
            )
        except Exception as exc:
            logger.warning("Telegram send failed: %s", mask_text(str(exc)))

    # ------------------------------------------------------------------
    # Typed alerts
    # ------------------------------------------------------------------

    def alert_entry(
        self,
        code: str,
        name: str,
        side: str,
        qty: int,
        price: float,
        stop: float,
        equity: float,
    ) -> None:
        pct = price * qty / equity * 100 if equity else 0.0
        self.send(
            f"📈 *진입* `{code} {name}`\n"
            f"방향: `{side}`  수량: `{qty:,}주`\n"
            f"가격: `{price:,.0f}`  손절: `{stop:,.0f}`\n"
            f"자산 대비: `{pct:.1f}%`"
        )

    def alert_exit(
        self,
        code: str,
        name: str,
        qty: int,
        price: float,
        entry_price: float,
        reason: str,
    ) -> None:
        pnl_pct = (price - entry_price) / entry_price * 100 if entry_price else 0.0
        sign = "+" if pnl_pct >= 0 else ""
        emoji = "✅" if pnl_pct >= 0 else "🔴"
        self.send(
            f"{emoji} *청산* `{code} {name}`\n"
            f"수량: `{qty:,}주`  가격: `{price:,.0f}`\n"
            f"수익률: `{sign}{pnl_pct:.2f}%`  사유: {reason}"
        )

    def alert_stop_hit(self, code: str, name: str, price: float, stop: float) -> None:
        loss_pct = (stop - price) / price * 100
        self.send(
            f"🚨 *손절 발동* `{code} {name}`\n"
            f"현재가: `{price:,.0f}`  손절가: `{stop:,.0f}`\n"
            f"손실폭: `{loss_pct:.2f}%`"
        )

    def alert_kill_switch(self, equity: float, peak: float) -> None:
        drawdown = (peak - equity) / peak * 100 if peak else 0.0
        self.send(
            f"⛔ *킬스위치 발동*\n"
            f"자산: `{equity:,.0f} KRW`  고점: `{peak:,.0f} KRW`\n"
            f"낙폭: `{drawdown:.1f}%`  → 모든 포지션 청산 후 정지."
        )

    def alert_paused(self, reason: str) -> None:
        self.send(f"⏸ *봇 일시정지*\n사유: {reason}\n`/resume`으로 재개.")

    def alert_resumed(self) -> None:
        self.send("▶️ *봇 재개* — 신규 진입 다시 허용.")

    def alert_close_all_done(self, closed: int) -> None:
        self.send(f"🔒 *전체 청산 완료* — {closed}개 포지션 닫음.")


# --------------------------------------------------------------------------
# Command handler
# --------------------------------------------------------------------------


class TelegramCommandHandler:
    """Poll getUpdates and dispatch /commands from the authorised chat."""

    HELP = (
        "*명령어 목록*\n"
        "`/status`   — 현재 포지션·잔고·손익\n"
        "`/stop`     — 신규 진입 중지 (포지션 관리는 계속)\n"
        "`/resume`   — 진입 재개\n"
        "`/close_all`— ⚠️ 전종목 강제 청산\n"
        "`/help`     — 이 도움말"
    )

    def __init__(
        self,
        token: str,
        chat_id: str,
        dry_run: bool = True,
        poll_interval: float = 3.0,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self.dry_run = dry_run
        self._poll_interval = poll_interval
        self._offset = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Callbacks — wired up by main.py after engine is built.
        self.on_status: Callable[[], str] = lambda: "⚠️ status 콜백 미설정"
        self.on_stop: Callable[[], str] = lambda: "⚠️ stop 콜백 미설정"
        self.on_resume: Callable[[], str] = lambda: "⚠️ resume 콜백 미설정"
        self.on_close_all: Callable[[], str] = lambda: "⚠️ close_all 콜백 미설정"

    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.dry_run or not self._token or not self._chat_id:
            logger.info("Telegram command handler: dry-run / no credentials — not starting")
            return
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="tg-cmd"
        )
        self._thread.start()
        logger.info(
            "Telegram command handler started (poll every %.0fs, chat_id=***)",
            self._poll_interval,
        )

    def shutdown(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _api(self, method: str, **kwargs: Any) -> dict[str, Any]:
        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{self._token}/{method}",
            json=kwargs,
            timeout=15.0,
        )
        return dict(resp.json())

    def _reply(self, text: str) -> None:
        try:
            self._api(
                "sendMessage",
                chat_id=self._chat_id,
                text=text[:_MAX_LEN],
                parse_mode="Markdown",
            )
        except Exception as exc:
            logger.warning("Telegram reply failed: %s", mask_text(str(exc)))

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._process_updates()
            except Exception as exc:
                logger.debug("Telegram poll error: %s", mask_text(str(exc)))
            time.sleep(self._poll_interval)

    def _process_updates(self) -> None:
        data = self._api("getUpdates", offset=self._offset, timeout=2)
        if not data.get("ok"):
            return
        for update in data.get("result", []):
            self._offset = int(update["update_id"]) + 1
            msg = update.get("message") or {}
            chat_id = str((msg.get("chat") or {}).get("id", ""))
            if chat_id != self._chat_id:
                # Silently ignore messages from other chats.
                continue
            text = str(msg.get("text", "")).strip()
            if not text:
                continue
            cmd = text.split()[0].lower().split("@")[0]  # strip @botname suffix
            self._dispatch(cmd)

    def _dispatch(self, cmd: str) -> None:
        logger.info("Telegram command received: %s", cmd)
        if cmd == "/help":
            self._reply(self.HELP)
        elif cmd == "/status":
            self._reply(self.on_status())
        elif cmd == "/stop":
            self._reply(self.on_stop())
        elif cmd == "/resume":
            self._reply(self.on_resume())
        elif cmd == "/close_all":
            logger.warning("Telegram /close_all received — executing emergency flatten")
            self._reply(self.on_close_all())
        else:
            self._reply(f"알 수 없는 명령어: `{cmd}`\n`/help`로 목록 확인")


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def build_telegram(
    credentials: Any,
    config: Any,
) -> tuple[TelegramNotifier, TelegramCommandHandler]:
    """Return (notifier, command_handler) from credentials + config."""
    tg_cfg = (config.get("telegram") or {}) if config else {}
    dry_run = bool(tg_cfg.get("dry_run", True))

    token = ""
    chat_id = ""
    if credentials is not None and getattr(credentials, "has_telegram", False):
        token = credentials.telegram_token.reveal()
        chat_id = credentials.telegram_chat_id.reveal()

    notifier = TelegramNotifier(token=token, chat_id=chat_id, dry_run=dry_run)
    handler = TelegramCommandHandler(
        token=token,
        chat_id=chat_id,
        dry_run=dry_run,
        poll_interval=float(tg_cfg.get("poll_interval_seconds", 3.0)),
    )
    return notifier, handler
