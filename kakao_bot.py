"""KakaoTalk alerts via the "나에게 보내기" (send-to-me) API.

2026-09-28, user request: replace Telegram entirely (실거래 알림을 텔레그램
대신 카카오톡으로). This is one-way push only -- there is no Kakao
equivalent of Telegram's getUpdates long-poll for a personal account
without a registered business channel, so the remote /stop, /resume,
/close_all commands that used to arrive over Telegram have no replacement
here. Use `python main.py stop`/`resume` from a terminal instead, or
Ctrl+C the process for an immediate full stop.

One-time setup (done by the user in a browser -- nothing here can do it,
there is no way to log into someone's Kakao account from a bot process):

  1. https://developers.kakao.com -> 내 애플리케이션 -> 애플리케이션 추가.
  2. 앱 설정 -> 카카오 로그인 -> 활성화 ON, Redirect URI 아무 값이나 등록
     (예: https://localhost.com).
  3. 카카오 로그인 -> 동의항목 -> "카카오톡 메시지 전송" 항목을 선택 가능으로 설정.
  4. 브라우저로 아래 주소에 접속 (<REST_API_KEY>, <REDIRECT_URI>는 위에서
     등록한 값으로 교체), 본인 계정으로 로그인:
     https://kauth.kakao.com/oauth/authorize?client_id=<REST_API_KEY>&redirect_uri=<REDIRECT_URI>&response_type=code
  5. 로그인 후 리다이렉트된 주소의 `code=` 뒤 값을 복사해서, 아래로 한 번
     교환한다 (터미널에서, <REST_API_KEY>/<REDIRECT_URI>/<복사한 code>를
     실제 값으로 교체):
     curl -X POST https://kauth.kakao.com/oauth/token -d grant_type=authorization_code -d client_id=<REST_API_KEY> -d redirect_uri=<REDIRECT_URI> -d code=<복사한 code>
     응답 JSON의 access_token/refresh_token 중 refresh_token만 저장하면 된다
     (access_token은 이 파일이 알아서 매번 새로 받아온다).
  6. .env에 KAKAO_REST_API_KEY(2번의 REST API 키)와
     KAKAO_REFRESH_TOKEN(5번의 refresh_token)을 채운다.

Token lifetime: an access_token lasts ~6 hours; this refreshes it
automatically from the refresh_token before every send once it's stale.
Kakao occasionally rotates the refresh_token itself on a refresh call --
when it does, the new one is written back into .env so a later restart
still works without redoing the browser login.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from settings import PROJECT_ROOT, mask_text

logger = logging.getLogger("bot.kakao")

_MAX_LEN = 190  # Kakao's default "text" template caps at 200 characters
_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
_SEND_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

_PREVIEW_REASON_DRY_RUN = "dry-run"
_PREVIEW_REASON_NO_CREDS = "no Kakao credentials configured"


class KakaoNotifier:
    """Push one-way alerts to the bot owner's own KakaoTalk.

    Exposes the same typed-alert surface TelegramNotifier used to (send,
    alert_entry, alert_exit, alert_stop_hit, alert_kill_switch), so it
    drops into TradingEngine._tg_notifier without touching any call site.
    """

    def __init__(
        self,
        rest_api_key: str,
        refresh_token: str,
        dry_run: bool = True,
        env_path: Path | None = None,
    ) -> None:
        self._rest_api_key = rest_api_key
        self._refresh_token = refresh_token
        self.dry_run = dry_run
        self._env_path = env_path or (PROJECT_ROOT / ".env")
        self._access_token = ""
        self._access_token_expiry = 0.0

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------

    def send(self, text: str) -> None:
        self.send_with_result(text)

    def send_with_result(self, text: str) -> dict[str, Any]:
        """Like send(), but returns {"sent", "reason", "preview"} -- used by
        report.py's CLI, which prints a preview/failure box either way."""
        # Kakao has no Markdown rendering -- strip the Telegram-era
        # emphasis markers call sites still pass so a message doesn't show
        # up littered with stray asterisks/backticks.
        text = re.sub(r"[*`_]", "", text)
        if self.dry_run:
            logger.debug("[Kakao preview] %s", text[:120])
            return {"sent": False, "reason": _PREVIEW_REASON_DRY_RUN, "preview": text}
        if not self._rest_api_key or not self._refresh_token:
            logger.debug("[Kakao preview] %s", text[:120])
            return {"sent": False, "reason": _PREVIEW_REASON_NO_CREDS, "preview": text}

        token = self._ensure_access_token()
        if not token:
            return {"sent": False, "reason": "token refresh failed", "preview": text}

        template = {
            "object_type": "text",
            "text": text[:_MAX_LEN],
            "link": {
                "web_url": "https://www.kiwoom.com",
                "mobile_web_url": "https://www.kiwoom.com",
            },
        }
        try:
            import requests

            response = self._post_memo(requests, token, template)
            if response.status_code == 401:
                # Access token expired earlier than its own expires_in said --
                # refresh once and retry rather than silently dropping the
                # alert.
                self._access_token = ""
                token = self._ensure_access_token()
                if not token:
                    return {"sent": False, "reason": "token refresh failed", "preview": text}
                response = self._post_memo(requests, token, template)
            response.raise_for_status()
            data = response.json()
            if data.get("result_code", 0) != 0:
                reason = mask_text(str(data))
                logger.warning("Kakao send failed: %s", reason)
                return {"sent": False, "reason": reason, "preview": text}
        except Exception as exc:
            reason = mask_text(str(exc))
            logger.warning("Kakao send failed: %s", reason)
            return {"sent": False, "reason": reason, "preview": text}

        logger.info("Kakao message delivered (%d chars)", len(text))
        return {"sent": True, "reason": "", "preview": text}

    def _post_memo(self, requests, token: str, template: dict[str, Any]):
        return requests.post(
            _SEND_URL,
            headers={"Authorization": f"Bearer {token}"},
            data={"template_object": json.dumps(template, ensure_ascii=False)},
            timeout=10.0,
        )

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    def _ensure_access_token(self) -> str:
        if self._access_token and time.time() < self._access_token_expiry:
            return self._access_token
        try:
            import requests

            resp = requests.post(
                _TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self._rest_api_key,
                    "refresh_token": self._refresh_token,
                },
                timeout=10.0,
            )
            data = resp.json()
            token = data.get("access_token")
            if not token:
                logger.warning(
                    "Kakao token refresh failed: %s",
                    mask_text(str(data.get("error_description", data))),
                )
                return ""
            self._access_token = token
            # 60s safety margin so a send started right at the edge of the
            # window never races an expiry mid-request.
            self._access_token_expiry = time.time() + int(data.get("expires_in", 21600)) - 60
            new_refresh = data.get("refresh_token")
            if new_refresh and new_refresh != self._refresh_token:
                self._refresh_token = new_refresh
                self._persist_refresh_token(new_refresh)
            return token
        except Exception as exc:
            logger.warning("Kakao token refresh failed: %s", mask_text(str(exc)))
            return ""

    def _persist_refresh_token(self, new_token: str) -> None:
        """Kakao rotates the refresh_token on some refreshes -- write the new
        one back into .env so the next restart doesn't need a fresh browser
        login. Best-effort: a failure here just means the user has to redo
        the one-time login once the old token finally expires.
        """
        try:
            path = self._env_path
            if not path.exists():
                return
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            out = []
            replaced = False
            for line in lines:
                if line.startswith("KAKAO_REFRESH_TOKEN="):
                    out.append(f"KAKAO_REFRESH_TOKEN={new_token}\n")
                    replaced = True
                else:
                    out.append(line)
            if not replaced:
                out.append(f"KAKAO_REFRESH_TOKEN={new_token}\n")
            path.write_text("".join(out), encoding="utf-8")
        except OSError as exc:
            logger.warning("Kakao refresh_token persist failed: %s", exc)

    # ------------------------------------------------------------------
    # Typed alerts (mirrors TelegramNotifier's surface)
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
            f"[진입] {code} {name}\n"
            f"방향: {side}  수량: {qty:,}주\n"
            f"가격: {price:,.0f}  손절: {stop:,.0f}\n"
            f"자산 대비: {pct:.1f}%"
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
        self.send(
            f"[청산] {code} {name}\n"
            f"수량: {qty:,}주  가격: {price:,.0f}\n"
            f"수익률: {sign}{pnl_pct:.2f}%  사유: {reason}"
        )

    def alert_stop_hit(self, code: str, name: str, price: float, stop: float) -> None:
        loss_pct = (stop - price) / price * 100
        self.send(
            f"[손절 발동] {code} {name}\n"
            f"현재가: {price:,.0f}  손절가: {stop:,.0f}\n"
            f"손실폭: {loss_pct:.2f}%"
        )

    def alert_kill_switch(self, equity: float, peak: float) -> None:
        drawdown = (peak - equity) / peak * 100 if peak else 0.0
        self.send(
            f"[킬스위치 발동]\n"
            f"자산: {equity:,.0f} KRW  고점: {peak:,.0f} KRW\n"
            f"낙폭: {drawdown:.1f}%  -> 모든 포지션 청산 후 정지."
        )


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def build_kakao(credentials: Any, config: Any) -> KakaoNotifier:
    """Return a KakaoNotifier from credentials + config."""
    kakao_cfg = (config.get("kakao") or {}) if config else {}
    dry_run = bool(kakao_cfg.get("dry_run", True))

    rest_api_key = ""
    refresh_token = ""
    if credentials is not None and getattr(credentials, "has_kakao", False):
        rest_api_key = credentials.kakao_rest_api_key.reveal()
        refresh_token = credentials.kakao_refresh_token.reveal()

    return KakaoNotifier(rest_api_key=rest_api_key, refresh_token=refresh_token, dry_run=dry_run)
