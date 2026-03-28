from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import requests

from src.config import settings
from src.data.fetchers.base import BasePriceFetcher

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.jquants.com/v1"

# 東証33業種コード → 名称マップ（主要のみ）
_SECTOR33_MAP: dict[str, str] = {
    "0050": "水産・農林業", "1050": "鉱業", "2050": "建設業",
    "3050": "食料品", "3100": "繊維製品", "3150": "パルプ・紙",
    "3200": "化学", "3250": "医薬品", "3300": "石油・石炭製品",
    "3350": "ゴム製品", "3400": "ガラス・土石製品", "3450": "鉄鋼",
    "3500": "非鉄金属", "3550": "金属製品", "3600": "機械",
    "3650": "電気機器", "3700": "輸送用機器", "3750": "精密機器",
    "3800": "その他製品", "4050": "電気・ガス業", "5050": "陸運業",
    "5100": "海運業", "5150": "空運業", "5200": "倉庫・運輸関連業",
    "5250": "情報・通信業", "6050": "卸売業", "6100": "小売業",
    "7050": "銀行業", "7100": "証券、商品先物取引業", "7150": "保険業",
    "7200": "その他金融業", "8050": "不動産業", "9050": "サービス業",
}

_MARKET_MAP: dict[str, str] = {
    "Prime": "Prime", "Standard": "Standard", "Growth": "Growth",
    "プライム": "Prime", "スタンダード": "Standard", "グロース": "Growth",
}


class JQuantsFetcher(BasePriceFetcher):
    """J-Quants API を使った株価データ取得"""

    def __init__(self) -> None:
        self._id_token: str | None = None
        self._refresh_token: str = settings.jquants_refresh_token

    # ------------------------------------------------------------------
    # 認証
    # ------------------------------------------------------------------
    def _get_refresh_token(self) -> str:
        if self._refresh_token:
            return self._refresh_token
        resp = requests.post(
            f"{_BASE_URL}/token/auth_user",
            json={"mailaddress": settings.jquants_mail, "password": settings.jquants_password},
            timeout=30,
        )
        resp.raise_for_status()
        token = resp.json()["refreshToken"]
        self._refresh_token = token
        return token

    def _get_id_token(self) -> str:
        if self._id_token:
            return self._id_token
        refresh = self._get_refresh_token()
        resp = requests.post(
            f"{_BASE_URL}/token/auth_refresh",
            params={"refreshtoken": refresh},
            timeout=30,
        )
        resp.raise_for_status()
        self._id_token = resp.json()["idToken"]
        return self._id_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._get_id_token()}"}

    def _get(self, path: str, params: dict | None = None) -> Any:
        resp = requests.get(f"{_BASE_URL}{path}", headers=self._headers(), params=params, timeout=60)
        if resp.status_code == 401:
            self._id_token = None
            resp = requests.get(f"{_BASE_URL}{path}", headers=self._headers(), params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # 銘柄一覧
    # ------------------------------------------------------------------
    def fetch_stock_list(self) -> list[dict[str, Any]]:
        try:
            data = self._get("/listed/info")
            items = data.get("info", [])
        except Exception as e:
            logger.error("J-Quants 銘柄一覧取得失敗: %s", e)
            return []

        result = []
        for item in items:
            ticker = item.get("Code", "")
            if not ticker:
                continue
            # J-Quants の Code は 4 or 5 桁数字 → yfinance 形式に変換
            ticker_yf = ticker[:4] + ".T" if not ticker.endswith(".T") else ticker
            market_raw = item.get("MarketCodeName", "")
            result.append(
                {
                    "ticker": ticker_yf,
                    "name": item.get("CompanyName", ticker_yf),
                    "name_en": item.get("CompanyNameEnglish"),
                    "sector_33": _SECTOR33_MAP.get(item.get("Sector33Code", ""), item.get("Sector33CodeName")),
                    "sector_17": item.get("Sector17CodeName"),
                    "market": _MARKET_MAP.get(market_raw, market_raw),
                    "lot_size": int(item.get("TradingUnit", 100)),
                }
            )
        logger.info("J-Quants: %d 銘柄取得", len(result))
        return result

    # ------------------------------------------------------------------
    # 株価
    # ------------------------------------------------------------------
    def fetch_prices(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for ticker in tickers:
            code = ticker.replace(".T", "")
            try:
                data = self._get(
                    "/prices/daily_quotes",
                    params={"code": code, "from": start_date.isoformat(), "to": end_date.isoformat()},
                )
                quotes = data.get("daily_quotes", [])
            except Exception as e:
                logger.warning("J-Quants 価格取得失敗 %s: %s", ticker, e)
                continue

            for q in quotes:
                adj = q.get("AdjustmentClose") or q.get("Close")
                if adj is None:
                    continue
                records.append(
                    {
                        "ticker": ticker,
                        "date": date.fromisoformat(q["Date"]),
                        "open": _safe(q.get("AdjustmentOpen") or q.get("Open")),
                        "high": _safe(q.get("AdjustmentHigh") or q.get("High")),
                        "low": _safe(q.get("AdjustmentLow") or q.get("Low")),
                        "close": _safe(q.get("Close")),
                        "adj_close": float(adj),
                        "volume": _safe_int(q.get("AdjustmentVolume") or q.get("Volume")),
                    }
                )
        logger.info("J-Quants: %d レコード取得完了 (%d 銘柄)", len(records), len(tickers))
        return records

    # ------------------------------------------------------------------
    # 財務指標
    # ------------------------------------------------------------------
    def fetch_financial_indicators(self, tickers: list[str]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for ticker in tickers:
            code = ticker.replace(".T", "")
            try:
                data = self._get("/fins/statements", params={"code": code})
                stmts = data.get("statements", [])
            except Exception as e:
                logger.warning("J-Quants 財務データ取得失敗 %s: %s", ticker, e)
                continue

            for s in stmts:
                fiscal_date_str = s.get("CurrentPeriodEndDate") or s.get("DisclosedDate")
                if not fiscal_date_str:
                    continue
                try:
                    fiscal_date = date.fromisoformat(fiscal_date_str[:10])
                except ValueError:
                    continue
                records.append(
                    {
                        "ticker": ticker,
                        "fiscal_date": fiscal_date,
                        "per": _safe(s.get("PER")),
                        "pbr": _safe(s.get("PBR")),
                        "roe": _safe(s.get("ROE")),
                        "equity_ratio": _safe(s.get("EquityToAssetRatio")),
                        "dividend_yield": _safe(s.get("DividendYield")),
                        "market_cap": _safe_int(s.get("MarketCapitalization")),
                        "revenue_growth": None,
                    }
                )
        return records


def _safe(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    try:
        return int(float(v)) if v is not None else None
    except (TypeError, ValueError):
        return None
