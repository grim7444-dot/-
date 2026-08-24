"""트럼프 관련 뉴스 모니터 (구글 뉴스 RSS, 참고용 알림 전용).

'트럼프' 헤드라인 중 무역·관세·산업 키워드가 함께 나오는 것만 감지해 텔레그램
으로 알린다.

이건 자동매매 신호가 아니다 -- 트럼프의 발언과 국내 개별 종목의 실제 주가
반응 사이엔 확실한 인과관계가 없고, 헤드라인 텍스트만으로 "어떤 종목이
오를지" 짚어내는 건 신뢰도가 낮다. 그래서 진입을 자동으로 막거나 부스트하지
않고, 관련 테마 키워드만 함께 보여줘 판단은 사람이 하도록 돕는다.

API키 없이 동작한다 (구글 뉴스 RSS는 공개 피드). 네트워크 오류나 피드 포맷
변경 시 조용히 넘어가고 봇은 계속 돈다.
"""

from __future__ import annotations

import logging
import threading
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


class NewsMonitor:
    """구글 뉴스 RSS를 폴링해 '트럼프' + 산업 키워드 헤드라인을 알린다."""

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("news") or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.poll_interval: float = float(cfg.get("poll_interval_seconds", 300))
        self.query: str = str(cfg.get("query", "트럼프"))
        self._seen: set[str] = set()
        self._notifier: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_notifier(self, notifier: Any) -> None:
        self._notifier = notifier

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

                logger.info("news: 트럼프 관련 뉴스 - %s [%s]", title, ", ".join(themes))
                if self._notifier:
                    try:
                        self._notifier.send(
                            f"트럼프 뉴스 감지\n{title}\n"
                            f"관련 테마: {', '.join(themes)}\n"
                            f"(참고용 -- 자동매매에 반영되지 않음)\n{link}"
                        )
                    except Exception as exc:
                        logger.debug("news: 텔레그램 전송 실패: %s", exc)
