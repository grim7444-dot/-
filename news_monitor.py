"""트럼프 관련 뉴스 모니터 (구글 뉴스 RSS) — 텔레그램 알림 + 진입 우선 부스트.

'트럼프' 헤드라인 중 무역·관세·산업 키워드가 함께 나오는 것만 감지해 텔레그램
으로 알리고, 매칭된 테마에 속한 종목(``config.yaml``의 ``themes:``)을
boost_duration_minutes 동안 "진입 우선" 상태로 만든다.

헤드라인 텍스트만으로는 뉴스가 호재인지 악재인지 판단할 수 없다 -- 감성분석
없이 "트럼프 + 관세"라는 키워드만 보고 매수를 걸면 사실상 동전 던지기다.
그래서 뉴스가 직접 주문을 내지는 않는다: 부스트는 DART 공시 부스트와 동일한
방식으로 "이 종목을 더 우선적으로 보라"는 신호일 뿐이고, 실제 진입은 항상
기존 전략(돌파/눌림목)의 기술적 신호가 떴을 때만 나간다.

API키 없이 동작한다 (구글 뉴스 RSS는 공개 피드). 네트워크 오류나 피드 포맷
변경 시 조용히 넘어가고 봇은 계속 돈다.
"""

from __future__ import annotations

import logging
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger("bot.news")

_THEME_KEYWORDS: dict[str, list[str]] = {
    "반도체/배터리": ["반도체", "배터리", "2차전지"],
    "관세/무역": ["관세", "무역", "수입", "수출", "제재"],
    "방위산업": ["방산", "국방", "무기"],
    "에너지": ["원유", "에너지", "태양광", "전력"],
    "제약/바이오": ["제약", "바이오", "백신"],
    "조선/철강": ["조선", "철강", "선박"],
}

# 뉴스 테마 -> config.yaml `themes:` 키. 매칭되면 그 테마 소속 종목을 부스트한다.
_NEWS_THEME_TO_CONFIG_THEME: dict[str, str] = {
    "반도체/배터리": "battery_materials",
    "에너지": "power_energy",
    "제약/바이오": "pharma",
    "조선/철강": "shipbuilding",
}


class NewsMonitor:
    """구글 뉴스 RSS를 폴링해 '트럼프' + 산업 키워드 헤드라인을 알린다."""

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("news") or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.poll_interval: float = float(cfg.get("poll_interval_seconds", 300))
        self.query: str = str(cfg.get("query", "트럼프"))
        self.boost_duration: float = float(cfg.get("boost_duration_minutes", 20)) * 60
        self._seen: set[str] = set()
        self._notifier: Any = None
        self._config_themes: dict[str, list[str]] = {}
        self._boosted: dict[str, float] = {}  # ticker -> 만료 monotonic
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_notifier(self, notifier: Any) -> None:
        self._notifier = notifier

    def set_theme_map(self, themes: dict[str, Any]) -> None:
        self._config_themes = {
            str(k): [str(c) for c in (v or [])] for k, v in themes.items()
        }

    def is_boosted(self, code: str) -> bool:
        exp = self._boosted.get(code)
        if exp is None:
            return False
        if time.monotonic() > exp:
            self._boosted.pop(code, None)
            return False
        return True

    def _boost_codes(self, news_themes: list[str]) -> set[str]:
        codes: set[str] = set()
        for nt in news_themes:
            key = _NEWS_THEME_TO_CONFIG_THEME.get(nt)
            if key and key in self._config_themes:
                codes.update(self._config_themes[key])
        return codes

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="news-monitor"
        )
        self._thread.start()
        logger.info(
            "news: 모니터 시작 (query=%r, interval=%ds)", self.query, self.poll_interval
        )

    def stop(self) -> None:
        self._stop.set()

    def _fetch(self) -> list[tuple[str, str]]:
        try:
            url = (
                f"https://news.google.com/rss/search?q={quote(self.query)}"
                "&hl=ko&gl=KR&ceid=KR:ko"
            )
            resp = requests.get(url, timeout=10)
            root = ET.fromstring(resp.content)
            items: list[tuple[str, str]] = []
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                if title:
                    items.append((title, link))
            return items
        except Exception as exc:
            logger.debug("news: 조회 실패: %s", exc)
            return []

    def _match_themes(self, title: str) -> list[str]:
        return [
            theme for theme, kws in _THEME_KEYWORDS.items()
            if any(kw in title for kw in kws)
        ]

    def _poll_loop(self) -> None:
        # 시작 시 기존 헤드라인은 알림 없이 흡수 -- 안 그러면 시작하자마자
        # 오늘 이전 뉴스 수십 건이 한꺼번에 텔레그램으로 쏟아진다.
        for title, _ in self._fetch():
            self._seen.add(title)
        logger.info("news: 초기화 완료 -- 기존 헤드라인 %d건 로드", len(self._seen))

        while not self._stop.wait(self.poll_interval):
            for title, link in self._fetch():
                if title in self._seen:
                    continue
                self._seen.add(title)

                themes = self._match_themes(title)
                if not themes:
                    continue  # 산업 키워드 없는 트럼프 뉴스는 건너뜀 (노이즈 방지)

                boost_codes = self._boost_codes(themes)
                expiry = time.monotonic() + self.boost_duration
                for code in boost_codes:
                    self._boosted[code] = expiry

                boost_note = (
                    f"\n진입 우선 부스트: {', '.join(sorted(boost_codes))} "
                    f"({self.boost_duration/60:.0f}분, 실제 진입은 기술적 신호 필요)"
                    if boost_codes else ""
                )
                logger.info(
                    "news: 트럼프 관련 뉴스 - %s [%s]%s",
                    title, ", ".join(themes),
                    f" -> 부스트 {sorted(boost_codes)}" if boost_codes else "",
                )
                if self._notifier:
                    try:
                        self._notifier.send(
                            f"트럼프 뉴스 감지\n{title}\n"
                            f"관련 테마: {', '.join(themes)}{boost_note}\n{link}"
                        )
                    except Exception as exc:
                        logger.debug("news: 텔레그램 전송 실패: %s", exc)
