"""
키움 OpenAPI+ 어댑터

요구사항:
  - 32비트 Python (3.8~3.11)
  - pip install pykiwoom PyQt5
  - 키움증권 영웅문 HTS + OpenAPI+ 설치/등록 완료
  - 실행 전 HTS 로그인 상태일 것

지원 심볼 형식:
  - 주식: "005930"  (삼성전자)
  - 코스닥: "035720" (카카오)
  - 선물: "101W9000" (KOSPI200 선물 근월물)
"""

import time
import pandas as pd
from loguru import logger
from .base import BaseExchange

try:
    from pykiwoom.kiwoom import Kiwoom
    _PYKIWOOM_AVAILABLE = True
except ImportError:
    _PYKIWOOM_AVAILABLE = False


# 주문 유형 상수
_ORDER_BUY  = 1
_ORDER_SELL = 2

# 호가 유형
_PRICE_MARKET = "03"   # 시장가
_PRICE_LIMIT  = "00"   # 지정가

# 분봉 타임프레임 → 분 단위
_TIMEFRAME_MIN = {
    "1m": 1, "3m": 3, "5m": 5, "10m": 10,
    "15m": 15, "30m": 30, "1h": 60,
}


class KiwoomExchange(BaseExchange):
    """키움 OpenAPI+ 기반 거래소 어댑터 (주식 + 선물)"""

    def __init__(self, account_no: str, dry_run: bool = True, market: str = "stock"):
        """
        Args:
            account_no: 키움 계좌번호 (예: "1234567890")
            dry_run: True면 주문을 실제로 내지 않음
            market: "stock" (주식) | "futures" (선물/옵션)
        """
        if not _PYKIWOOM_AVAILABLE:
            raise ImportError(
                "pykiwoom이 설치되지 않았습니다.\n"
                "32비트 Python 환경에서: pip install pykiwoom PyQt5"
            )

        self.account_no = account_no
        self.dry_run = dry_run
        self.market = market

        logger.info("키움 OpenAPI 로그인 중...")
        self._kw = Kiwoom()
        self._kw.CommConnect(block=True)
        logger.info(f"로그인 완료 | 계좌: {account_no} | 시장: {market} | dry_run={dry_run}")

    # ------------------------------------------------------------------
    # OHLCV 조회
    # ------------------------------------------------------------------

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        if timeframe == "1d":
            return self._get_daily(symbol, limit)
        elif timeframe in _TIMEFRAME_MIN:
            return self._get_minute(symbol, _TIMEFRAME_MIN[timeframe], limit)
        else:
            raise ValueError(f"지원하지 않는 타임프레임: {timeframe}. 사용 가능: 1m 3m 5m 10m 15m 30m 1h 1d")

    def _get_daily(self, symbol: str, limit: int) -> pd.DataFrame:
        """일봉 조회 — opt10081 (주식) / opt50001 (선물)"""
        if self.market == "futures":
            tr = "opt50001"
            input_vals = {"선물종목코드": symbol}
            cols_map = {
                "일자": "date", "시가": "open", "고가": "high",
                "저가": "low", "현재가": "close", "거래량": "volume",
            }
        else:
            tr = "opt10081"
            input_vals = {"종목코드": symbol, "기준일자": "", "수정주가구분": "1"}
            cols_map = {
                "일자": "date", "시가": "open", "고가": "high",
                "저가": "low", "현재가": "close", "거래량": "volume",
            }

        df = self._kw.block_request(
            tr, output="주식일봉차트조회", next=0, **input_vals
        )
        return self._normalize(df, cols_map, limit, date_col="date")

    def _get_minute(self, symbol: str, tick: int, limit: int) -> pd.DataFrame:
        """분봉 조회 — opt10080 (주식) / opt50002 (선물)"""
        if self.market == "futures":
            tr = "opt50002"
            input_vals = {"선물종목코드": symbol, "틱범위": str(tick)}
            date_col = "체결시간"
        else:
            tr = "opt10080"
            input_vals = {"종목코드": symbol, "틱범위": str(tick), "수정주가구분": "1"}
            date_col = "체결시간"

        cols_map = {
            date_col: "date", "시가": "open", "고가": "high",
            "저가": "low", "현재가": "close", "거래량": "volume",
        }
        df = self._kw.block_request(
            tr, output="주식분봉차트조회", next=0, **input_vals
        )
        return self._normalize(df, cols_map, limit, date_col="date")

    @staticmethod
    def _normalize(df: pd.DataFrame, cols_map: dict, limit: int, date_col: str) -> pd.DataFrame:
        df = df.rename(columns=cols_map)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce").abs()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col).sort_index()
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df.tail(limit)

    # ------------------------------------------------------------------
    # 잔고/계좌 조회
    # ------------------------------------------------------------------

    def get_balance(self) -> dict:
        df = self._kw.block_request(
            "opw00001",
            계좌번호=self.account_no,
            비밀번호="",
            비밀번호입력매체구분="00",
            조회구분="2",
            output="예수금상세현황",
            next=0,
        )
        try:
            deposit = int(str(df["예수금"].iloc[0]).replace(",", ""))
        except Exception:
            deposit = 0
        return {"total": {"KRW": float(deposit)}}

    def get_ticker(self, symbol: str) -> dict:
        price = self._kw.GetMasterLastPrice(symbol)
        price = abs(int(str(price).replace(",", "")))
        return {"symbol": symbol, "last": float(price)}

    # ------------------------------------------------------------------
    # 주문
    # ------------------------------------------------------------------

    def create_market_buy(self, symbol: str, amount: float) -> dict:
        return self._send_order(symbol, _ORDER_BUY, int(amount), 0, _PRICE_MARKET)

    def create_market_sell(self, symbol: str, amount: float) -> dict:
        return self._send_order(symbol, _ORDER_SELL, int(amount), 0, _PRICE_MARKET)

    def create_limit_buy(self, symbol: str, amount: float, price: float) -> dict:
        return self._send_order(symbol, _ORDER_BUY, int(amount), int(price), _PRICE_LIMIT)

    def create_limit_sell(self, symbol: str, amount: float, price: float) -> dict:
        return self._send_order(symbol, _ORDER_SELL, int(amount), int(price), _PRICE_LIMIT)

    def _send_order(self, symbol: str, order_type: int, qty: int, price: int, price_type: str) -> dict:
        side = "매수" if order_type == _ORDER_BUY else "매도"
        price_label = "시장가" if price_type == _PRICE_MARKET else f"{price:,}원"

        if self.dry_run:
            logger.info(f"[DRY RUN] {side} {symbol} {qty}주 @ {price_label}")
            return {"id": "dry-run", "symbol": symbol, "side": side, "qty": qty, "price": price}

        result = self._kw.SendOrder(
            "자동매매주문",
            "0101",          # 화면번호
            self.account_no,
            order_type,
            symbol,
            qty,
            price,
            price_type,
            "",
        )
        if result != 0:
            raise RuntimeError(f"주문 실패 (에러코드: {result})")

        logger.info(f"{side} 주문 전송: {symbol} {qty}주 @ {price_label}")
        return {"id": "submitted", "symbol": symbol, "side": side, "qty": qty, "price": price}

    # ------------------------------------------------------------------
    # 미체결 / 취소
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        logger.warning(f"키움 주문 취소는 별도 구현 필요: {order_id}")
        return {"id": order_id}

    def get_open_orders(self, symbol: str) -> list:
        return []
