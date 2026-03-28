from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.deps import get_db, get_price_data, get_stock_info, get_universe_tickers
from src.api.schemas.request import RecommendRequest
from src.api.schemas.response import (
    PortfolioItemResponse,
    PortfolioMetrics,
    RecommendResponse,
)
from src.services.recommend import RecommendService

router = APIRouter(prefix="/recommend", tags=["recommend"])
logger = logging.getLogger(__name__)


@router.post("/portfolio", response_model=RecommendResponse)
def recommend_portfolio(
    req: RecommendRequest,
    db: Session = Depends(get_db),
) -> RecommendResponse:
    # 銘柄ユニバース取得
    tickers = get_universe_tickers(
        sectors=req.filters.sectors,
        markets=req.filters.markets,
        max_stocks=req.filters.max_stocks,
        db=db,
    )

    # 価格データ取得
    price_df = get_price_data(tickers, req.lookback_days, db)
    stock_info = get_stock_info(tickers, db)

    # 財務フィルタ（市場時価総額・流動性）
    if req.filters.min_avg_volume is not None:
        from src.data.repository import PriceRepository
        from datetime import date
        price_repo = PriceRepository(db)
        liquid = price_repo.get_liquid_tickers(
            min_avg_volume=req.filters.min_avg_volume,
            lookback_days=20,
        )
        if liquid:
            price_df = price_df[[t for t in price_df.columns if t in liquid]]

    if price_df.shape[1] < 2:
        raise HTTPException(status_code=422, detail="条件を満たす銘柄が不足しています（最低 2 銘柄）")

    svc = RecommendService(
        lookback_days=req.lookback_days,
        risk_free_rate=req.risk_free_rate,
    )

    try:
        result = svc.recommend(
            price_df=price_df,
            stock_info=stock_info,
            risk_tolerance=req.risk_tolerance,
            investment_amount=req.investment_amount,
            max_weight_per_stock=req.constraints.max_weight_per_stock,
            max_weight_per_sector=req.constraints.max_weight_per_sector,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("推薦計算エラー")
        raise HTTPException(status_code=500, detail=f"最適化エラー: {e}")

    items = [
        PortfolioItemResponse(
            ticker=item.ticker,
            name=item.name,
            sector=item.sector,
            weight=item.weight,
            shares=item.shares,
            amount=item.amount,
            expected_return_contribution=item.expected_return_contribution,
        )
        for item in result.portfolio
    ]

    metrics_data = result.metrics
    metrics = PortfolioMetrics(
        expected_annual_return=metrics_data.get("expected_annual_return", 0.0),
        annual_volatility=metrics_data.get("annual_volatility", 0.0),
        sharpe_ratio=metrics_data.get("sharpe_ratio", 0.0),
        var_95=metrics_data.get("var_95"),
        cvar_95=metrics_data.get("cvar_95"),
    )

    return RecommendResponse(
        portfolio=items,
        metrics=metrics,
        total_invested=result.total_invested,
        cash_remaining=result.cash_remaining,
        computed_at=result.computed_at,
        optimization_method=result.optimization_method,
        warning=result.warning,
    )
