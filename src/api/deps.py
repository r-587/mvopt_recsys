from __future__ import annotations

"""FastAPI 依存性注入ユーティリティ"""

import logging
from datetime import date, timedelta
from typing import Annotated

import pandas as pd
import yfinance as yf
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from src.config import settings
from src.data.database import get_db
from src.data.fetchers.yfinance import YFinanceFetcher
from src.data.models import Stock
from src.data.repository import PriceRepository, StockRepository

logger = logging.getLogger(__name__)


def get_price_data(
    tickers: list[str],
    lookback_days: int,
    db: Session,
) -> pd.DataFrame:
    """DB またはフォールバックで価格データを取得"""
    end = date.today()
    start = end - timedelta(days=lookback_days * 2)  # 欠損考慮で多めに取得

    repo = PriceRepository(db)
    price_df = repo.get_prices(tickers, start, end)

    if price_df.empty or len(price_df) < lookback_days // 4:
        logger.info("DB にデータ不足。yfinance からフォールバック取得")
        fetcher = YFinanceFetcher()
        records = fetcher.fetch_prices(tickers, start, end)
        if records:
            data: dict[str, dict] = {}
            for r in records:
                t = r["ticker"]
                d = r["date"]
                if t not in data:
                    data[t] = {}
                data[t][d] = r["adj_close"]
            price_df = pd.DataFrame(data)

    if price_df.empty:
        raise HTTPException(status_code=422, detail="価格データを取得できませんでした")

    return price_df.tail(lookback_days * 2)


def get_stock_info(tickers: list[str], db: Session) -> pd.DataFrame:
    """銘柄情報を取得"""
    try:
        repo = StockRepository(db)
        stocks = repo.get_all_active()
        if not stocks:
            return pd.DataFrame()
        rows = [
            {
                "ticker": s.ticker,
                "name": s.name,
                "sector_33": s.sector_33,
                "market": s.market,
                "lot_size": s.lot_size,
            }
            for s in stocks
            if s.ticker in tickers
        ]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("ticker")
        return df
    except Exception:
        return pd.DataFrame()


def get_universe_tickers(
    sectors: list[str] | None,
    markets: list[str] | None,
    max_stocks: int,
    db: Session,
) -> list[str]:
    """フィルタ条件に合致する銘柄コードリストを返す"""
    try:
        repo = StockRepository(db)
        stocks = repo.get_universe(markets=markets, sectors=sectors)
        tickers = [s.ticker for s in stocks[:max_stocks]]
    except Exception:
        tickers = []

    if not tickers:
        tickers = YFinanceFetcher.default_tickers()[:max_stocks]

    return tickers
