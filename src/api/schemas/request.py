from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class StockFilter(BaseModel):
    sectors: list[str] | None = None
    markets: list[str] | None = Field(default=None, description="Prime / Standard / Growth")
    min_market_cap: int | None = Field(default=None, ge=0)
    min_avg_volume: int | None = Field(default=None, ge=0)
    max_stocks: int = Field(default=50, ge=2, le=200)


class PortfolioConstraints(BaseModel):
    max_weight_per_stock: float = Field(default=0.3, ge=0.01, le=1.0)
    max_weight_per_sector: float = Field(default=0.5, ge=0.01, le=1.0)


class RecommendRequest(BaseModel):
    investment_amount: int = Field(default=1_000_000, ge=100_000)
    risk_tolerance: Literal["low", "medium", "high"] = "medium"
    filters: StockFilter = Field(default_factory=StockFilter)
    constraints: PortfolioConstraints = Field(default_factory=PortfolioConstraints)
    lookback_days: int = Field(default=252, ge=60, le=1260)
    risk_free_rate: float = Field(default=0.001, ge=0.0, le=0.1)


class EfficientFrontierRequest(BaseModel):
    tickers: list[str] | None = None
    filters: StockFilter = Field(default_factory=StockFilter)
    n_points: int = Field(default=50, ge=10, le=200)
    lookback_days: int = Field(default=252, ge=60, le=1260)
    risk_free_rate: float = Field(default=0.001, ge=0.0, le=0.1)
    max_weight_per_stock: float = Field(default=0.3, ge=0.01, le=1.0)


class OptimizeRequest(BaseModel):
    tickers: list[str] | None = None
    filters: StockFilter = Field(default_factory=StockFilter)
    lookback_days: int = Field(default=252, ge=60, le=1260)
    risk_free_rate: float = Field(default=0.001, ge=0.0, le=0.1)
    max_weight_per_stock: float = Field(default=0.3, ge=0.01, le=1.0)
    target_return: float | None = Field(default=None, ge=0.0, le=1.0)


class BacktestRequest(BaseModel):
    tickers: list[str] | None = None
    filters: StockFilter = Field(default_factory=StockFilter)
    start_date: date
    end_date: date
    rebalance_freq: Literal["monthly", "quarterly", "annually"] = "monthly"
    lookback_days: int = Field(default=252, ge=60, le=1260)
    transaction_cost: float = Field(default=0.001, ge=0.0, le=0.05)
    max_weight_per_stock: float = Field(default=0.3, ge=0.01, le=1.0)
    include_benchmark: bool = True
