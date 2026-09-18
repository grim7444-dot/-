"""DART 전자공시 실시간 모니터.

금융감독원 DART OpenAPI를 poll_interval_seconds마다 폴링해 유니버스 종목의
신규 공시를 감지한다. 긍정 공시(계약·실적)는 해당 종목을 boost_duration_minutes간
"진입 우선" 상태로 만들고 텔레그램 알림을 보낸다. 부정 공시(CB·유상증자)는
당일 진입을 차단하고 텔레그램 알림을 보낸다.

설정:
  1. https://opendart.fss.or.kr/ 에서 무료 API키 발급
  2. .env 에 DART_API_KEY=발급받은키 추가
  3. config.yaml:
       dart:
         enabled: true
         poll_interval_seconds: 60
         boost_duration_minutes: 30

API키가 없으면 enabled: true여도 경고만 내고 모니터를 실행하지 않는다.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date
from typing import Any

import requests

logger = logging.getLogger("bot.dart")

_BULLISH_KEYWORDS = [
    "단일판매",
    "공급계약체결",
    "주요계약체결",
    "영업(잠정)실적",
    "잠정실적",
    "수주",
]

_BEARISH_KEYWORDS = [
    "전환사채",
    "신주인수권부사채",
    "유상증자",
]


class DartDisclosure:
    def __init__(self, data: dict[str, Any]) -> None:
        self.corp_name: str = data.get("corp_name", "")
        self.stock_code: str = data.get("stock_code", "")
        self.report_nm: str = data.get("report_nm", "")
        self.rcept_no: str = data.get("rcept_no", "")

    @property
    def url(self) -> str:
        return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={self.rcept_no}"

    @property
    def is_bullish(self) -> bool:
        return any(kw in self.report_nm for kw in _BULLISH_KEYWORDS)

    @property
    def is_bearish(self) -> bool:
        return any(kw in self.report_nm for kw in _BEARISH_KEYWORDS)


class DartMonitor:
    """유니버스 종목의 DART 공시를 실시간으로 감시한다."""

    def __init__(self, config: dict[str, Any], credentials: dict[str, Any]) -> None:
        cfg = config.get("dart") or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.poll_interval: float = float(cfg.get("poll_interval_seconds", 60))
        self.boost_duration: float = float(cfg.get("boost_duration_minutes", 30)) * 60
        self._api_key: str = str(credentials.get("dart_api_key") or "")

        self._boosted: dict[str, float] = {}   # ticker → 만료 monotonic
        self._bearish_codes: set[str] = set()  # 당일 차단 종목
        self._seen: set[str] = set()           # 처리한 rcept_no
        self._universe: set[str] = set()
        self._notifier: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_universe(self, tickers: set[str]) -> None:
        self._universe = tickers

    def set_notifier(self, notifier: Any) -> None:
        self._notifier = notifier

    def start(self) -> None:
        if not self.enabled:
            return
        if not self._api_key:
            logger.warning("dart: enabled이지만 DART_API_KEY 미설정 — 모니터 비활성")
            return
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="dart-monitor"
        )
        self._thread.start()
        logger.info("dart: 모니터 시작 (interval=%ds)", self.poll_interval)

    def stop(self) -> None:
        self._stop.set()

    def is_boosted(self, ticker: str) -> bool:
        exp = self._boosted.get(ticker)
        if exp is None:
            return False
        if time.monotonic() > exp:
            self._boosted.pop(ticker, None)
            return False
        return True

    def is_bearish_blocked(self, ticker: str) -> bool:
        return ticker in self._bearish_codes

    def _fetch(self, bgn_de: str) -> list[dict[str, Any]]:
        try:
            resp = requests.get(
                "https://opendart.fss.or.kr/api/list.json",
                params={
                    "crtfc_key": self._api_key,
                    "bgn_de": bgn_de,
                    "sort": "date",
                    "sort_mth": "desc",
                    "page_count": 100,
                },
                timeout=10,
            )
            data = resp.json()
            if data.get("status") != "000":
                logger.debug(
                    "dart: API 상태 %s: %s",
                    data.get("status"), data.get("message"),
                )
                return []
            return data.get("list", [])
        except Exception as exc:
            logger.debug("dart: 조회 실패: %s", exc)
            return []

    def _poll_loop(self) -> None:
        today = date.today().strftime("%Y%m%d")
        # 시작 시 기존 공시 수집 (알림 없이) → 이전 공시로 오알림 방지
        for item in self._fetch(today):
            self._seen.add(item.get("rcept_no", ""))
        logger.info("dart: 초기화 완료 — 기존 공시 %d건 로드", len(self._seen))

        while not self._stop.wait(self.poll_interval):
            today = date.today().strftime("%Y%m%d")
            for item in self._fetch(today):
                rcept_no = item.get("rcept_no", "")
                if rcept_no in self._seen:
                    continue
                self._seen.add(rcept_no)

                disc = DartDisclosure(item)
                if not disc.stock_code or disc.stock_code not in self._universe:
                    continue

                logger.info(
                    "dart: 새 공시 %s %s — %s",
                    disc.stock_code, disc.corp_name, disc.report_nm,
                )

                if disc.is_bearish:
                    self._bearish_codes.add(disc.stock_code)
                    msg = (
                        f"공시 주의 {disc.corp_name}({disc.stock_code})\n"
                        f"{disc.report_nm}\n진입 차단됨\n{disc.url}"
                    )
                elif disc.is_bullish:
                    self._boosted[disc.stock_code] = time.monotonic() + self.boost_duration
                    msg = (
                        f"공시 감지 {disc.corp_name}({disc.stock_code})\n"
                        f"{disc.report_nm}\n진입 우선 ({self.boost_duration/60:.0f}분)\n{disc.url}"
                    )
                else:
                    msg = (
                        f"공시 {disc.corp_name}({disc.stock_code})\n"
                        f"{disc.report_nm}"
                    )

                if self._notifier:
                    try:
                        self._notifier.send(msg)
                    except Exception as exc:
                        logger.debug("dart: 텔레그램 전송 실패: %s", exc)
