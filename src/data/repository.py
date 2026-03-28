from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import and_, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.config import settings
from src.data.models import DailyPrice, FinancialIndicator, RecommendationHistory, Stock


class StockRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_all_active(self) -> list[Stock]:
        stmt = select(Stock).where(Stock.is_active.is_(True))
        return list(self._db.scalars(stmt))

    def get_universe(
        self,
        markets: list[str] | None = None,
        sectors: list[str] | None = None,
    ) -> list[Stock]:
        stmt = select(Stock).where(Stock.is_active.is_(True))
        if markets:
            stmt = stmt.where(Stock.market.in_(markets))
        if sectors:
            stmt = stmt.where(Stock.sector_33.in_(sectors))
        return list(self._db.scalars(stmt))

    def upsert_stocks(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        stmt = insert(Stock).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker"],
            set_={c: stmt.excluded[c] for c in ["name", "sector_33", "sector_17", "market", "lot_size", "is_active"]},
        )
        self._db.execute(stmt)
        self._db.commit()


class PriceRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_prices(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        stmt = (
            select(DailyPrice)
            .where(
                and_(
                    DailyPrice.ticker.in_(tickers),
                    DailyPrice.date >= start_date,
                    DailyPrice.date <= end_date,
                )
            )
            .order_by(DailyPrice.date)
        )
        rows = self._db.scalars(stmt).all()
        if not rows:
            return pd.DataFrame()
        data = [
            {"ticker": r.ticker, "date": r.date, "adj_close": float(r.adj_close), "volume": r.volume}
            for r in rows
        ]
        df = pd.DataFrame(data)
        return df.pivot(index="date", columns="ticker", values="adj_close")

    def get_volume_pivot(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        stmt = (
            select(DailyPrice)
            .where(
                and_(
                    DailyPrice.ticker.in_(tickers),
                    DailyPrice.date >= start_date,
                    DailyPrice.date <= end_date,
                )
            )
            .order_by(DailyPrice.date)
        )
        rows = self._db.scalars(stmt).all()
        if not rows:
            return pd.DataFrame()
        data = [
            {"ticker": r.ticker, "date": r.date, "volume": r.volume or 0}
            for r in rows
        ]
        df = pd.DataFrame(data)
        return df.pivot(index="date", columns="ticker", values="volume")

    def get_latest_prices(self, tickers: list[str]) -> dict[str, float]:
        subq = (
            select(DailyPrice.ticker, DailyPrice.adj_close)
            .where(DailyPrice.ticker.in_(tickers))
            .order_by(DailyPrice.ticker, DailyPrice.date.desc())
            .distinct(DailyPrice.ticker)
        )
        rows = self._db.execute(subq).all()
        return {row.ticker: float(row.adj_close) for row in rows}

    def upsert_prices(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        stmt = insert(DailyPrice).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "date"],
            set_={c: stmt.excluded[c] for c in ["open", "high", "low", "close", "adj_close", "volume"]},
        )
        self._db.execute(stmt)
        self._db.commit()

    def get_liquid_tickers(
        self,
        min_avg_volume: int,
        lookback_days: int,
        reference_date: date | None = None,
    ) -> list[str]:
        if reference_date is None:
            reference_date = date.today()
        start = reference_date - timedelta(days=lookback_days * 2)
        query = text(
            """
            SELECT ticker
            FROM daily_prices
            WHERE date >= :start AND date <= :end
            GROUP BY ticker
            HAVING COUNT(*) >= :min_days AND AVG(volume) >= :min_vol
            """
        )
        rows = self._db.execute(
            query,
            {"start": start, "end": reference_date, "min_days": lookback_days // 2, "min_vol": min_avg_volume},
        ).all()
        return [row[0] for row in rows]


class FinancialRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_latest_indicators(self, tickers: list[str]) -> pd.DataFrame:
        subq = (
            select(FinancialIndicator)
            .where(FinancialIndicator.ticker.in_(tickers))
            .order_by(FinancialIndicator.ticker, FinancialIndicator.fiscal_date.desc())
            .distinct(FinancialIndicator.ticker)
        )
        rows = self._db.scalars(subq).all()
        if not rows:
            return pd.DataFrame()
        data = [
            {
                "ticker": r.ticker,
                "per": r.per,
                "pbr": r.pbr,
                "roe": r.roe,
                "equity_ratio": r.equity_ratio,
                "dividend_yield": r.dividend_yield,
                "market_cap": r.market_cap,
                "revenue_growth": r.revenue_growth,
            }
            for r in rows
        ]
        return pd.DataFrame(data).set_index("ticker")

    def upsert_indicators(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        stmt = insert(FinancialIndicator).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "fiscal_date"],
            set_={
                c: stmt.excluded[c]
                for c in ["per", "pbr", "roe", "equity_ratio", "dividend_yield", "market_cap", "revenue_growth"]
            },
        )
        self._db.execute(stmt)
        self._db.commit()


class RecommendationRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def save(
        self,
        risk_tolerance: str,
        result_json: dict[str, Any],
        investment_amt: int | None = None,
        expected_return: float | None = None,
        expected_vol: float | None = None,
        sharpe_ratio: float | None = None,
        user_id: str | None = None,
    ) -> int:
        rec = RecommendationHistory(
            user_id=user_id,
            risk_tolerance=risk_tolerance,
            investment_amt=investment_amt,
            result_json=result_json,
            expected_return=expected_return,
            expected_vol=expected_vol,
            sharpe_ratio=sharpe_ratio,
        )
        self._db.add(rec)
        self._db.commit()
        self._db.refresh(rec)
        return rec.id

    def get_history(self, user_id: str | None = None, limit: int = 20) -> list[RecommendationHistory]:
        stmt = select(RecommendationHistory).order_by(RecommendationHistory.created_at.desc()).limit(limit)
        if user_id:
            stmt = stmt.where(RecommendationHistory.user_id == user_id)
        return list(self._db.scalars(stmt))
