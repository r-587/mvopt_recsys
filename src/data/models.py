from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Stock(Base):
    __tablename__ = "stocks"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    name_en: Mapped[str | None] = mapped_column(String(100))
    sector_33: Mapped[str | None] = mapped_column(String(50))
    sector_17: Mapped[str | None] = mapped_column(String(50))
    market: Mapped[str | None] = mapped_column(String(20))
    lot_size: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class DailyPrice(Base):
    __tablename__ = "daily_prices"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float | None] = mapped_column(Numeric(12, 2))
    high: Mapped[float | None] = mapped_column(Numeric(12, 2))
    low: Mapped[float | None] = mapped_column(Numeric(12, 2))
    close: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    adj_close: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    volume: Mapped[int | None] = mapped_column(BigInteger)


class FinancialIndicator(Base):
    __tablename__ = "financial_indicators"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    fiscal_date: Mapped[date] = mapped_column(Date, primary_key=True)
    per: Mapped[float | None] = mapped_column(Numeric(10, 2))
    pbr: Mapped[float | None] = mapped_column(Numeric(10, 2))
    roe: Mapped[float | None] = mapped_column(Numeric(10, 4))
    equity_ratio: Mapped[float | None] = mapped_column(Numeric(10, 4))
    dividend_yield: Mapped[float | None] = mapped_column(Numeric(10, 4))
    market_cap: Mapped[int | None] = mapped_column(BigInteger)
    revenue_growth: Mapped[float | None] = mapped_column(Numeric(10, 4))


class RecommendationHistory(Base):
    __tablename__ = "recommendation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    user_id: Mapped[str | None] = mapped_column(String(50))
    risk_tolerance: Mapped[str] = mapped_column(String(10), nullable=False)
    investment_amt: Mapped[int | None] = mapped_column(BigInteger)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    expected_return: Mapped[float | None] = mapped_column(Numeric(10, 4))
    expected_vol: Mapped[float | None] = mapped_column(Numeric(10, 4))
    sharpe_ratio: Mapped[float | None] = mapped_column(Numeric(10, 4))
