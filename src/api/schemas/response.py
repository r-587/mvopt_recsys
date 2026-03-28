from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel


class PortfolioItemResponse(BaseModel):
    ticker: str
    name: str
    sector: str | None
    weight: float
    shares: int
    amount: int
    expected_return_contribution: float


class PortfolioMetrics(BaseModel):
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    var_95: float | None = None
    cvar_95: float | None = None


class RecommendResponse(BaseModel):
    portfolio: list[PortfolioItemResponse]
    metrics: PortfolioMetrics
    total_invested: int
    cash_remaining: int
    computed_at: date
    optimization_method: str
    warning: str | None = None


class FrontierPoint(BaseModel):
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: dict[str, float]


class EfficientFrontierResponse(BaseModel):
    frontier_points: list[FrontierPoint]
    max_sharpe_point: dict[str, float]
    min_vol_point: dict[str, float]
    tickers: list[str]


class OptimizeResponse(BaseModel):
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    status: str
    method: str


class BacktestResponse(BaseModel):
    portfolio_values: dict[str, float]
    benchmark_values: dict[str, float] | None
    rebalance_dates: list[str]
    metrics: dict[str, float]
    tickers: list[str]


class CorrelationResponse(BaseModel):
    matrix: dict[str, dict[str, float]]
    tickers: list[str]


class StockInfo(BaseModel):
    ticker: str
    name: str
    sector_33: str | None
    market: str | None
    lot_size: int


class UniverseResponse(BaseModel):
    stocks: list[StockInfo]
    total: int


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
