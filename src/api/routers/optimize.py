from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.deps import get_db, get_price_data, get_universe_tickers
from src.api.schemas.request import EfficientFrontierRequest, OptimizeRequest
from src.api.schemas.response import (
    EfficientFrontierResponse,
    FrontierPoint,
    OptimizeResponse,
)
from src.services.optimize import OptimizeService

router = APIRouter(prefix="/optimize", tags=["optimize"])
logger = logging.getLogger(__name__)


def _resolve_tickers(req_tickers, filters, db) -> list[str]:
    if req_tickers:
        return req_tickers
    return get_universe_tickers(
        sectors=filters.sectors,
        markets=filters.markets,
        max_stocks=filters.max_stocks,
        db=db,
    )


@router.post("/max-sharpe", response_model=OptimizeResponse)
def max_sharpe(
    req: OptimizeRequest,
    db: Session = Depends(get_db),
) -> OptimizeResponse:
    tickers = _resolve_tickers(req.tickers, req.filters, db)
    price_df = get_price_data(tickers, req.lookback_days, db)
    svc = OptimizeService(lookback_days=req.lookback_days, risk_free_rate=req.risk_free_rate)
    try:
        res = svc.max_sharpe(price_df, weight_bounds=(0.0, req.max_weight_per_stock))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return OptimizeResponse(
        weights=res.weights_dict,
        expected_return=res.expected_return,
        volatility=res.volatility,
        sharpe_ratio=res.sharpe_ratio,
        status=res.status,
        method="max_sharpe",
    )


@router.post("/min-volatility", response_model=OptimizeResponse)
def min_volatility(
    req: OptimizeRequest,
    db: Session = Depends(get_db),
) -> OptimizeResponse:
    tickers = _resolve_tickers(req.tickers, req.filters, db)
    price_df = get_price_data(tickers, req.lookback_days, db)
    svc = OptimizeService(lookback_days=req.lookback_days, risk_free_rate=req.risk_free_rate)
    try:
        res = svc.min_volatility(price_df, weight_bounds=(0.0, req.max_weight_per_stock))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return OptimizeResponse(
        weights=res.weights_dict,
        expected_return=res.expected_return,
        volatility=res.volatility,
        sharpe_ratio=res.sharpe_ratio,
        status=res.status,
        method="min_volatility",
    )


@router.post("/target-return", response_model=OptimizeResponse)
def target_return(
    req: OptimizeRequest,
    db: Session = Depends(get_db),
) -> OptimizeResponse:
    if req.target_return is None:
        raise HTTPException(status_code=422, detail="target_return を指定してください")
    tickers = _resolve_tickers(req.tickers, req.filters, db)
    price_df = get_price_data(tickers, req.lookback_days, db)
    svc = OptimizeService(lookback_days=req.lookback_days, risk_free_rate=req.risk_free_rate)
    try:
        res = svc.target_return(price_df, req.target_return, weight_bounds=(0.0, req.max_weight_per_stock))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return OptimizeResponse(
        weights=res.weights_dict,
        expected_return=res.expected_return,
        volatility=res.volatility,
        sharpe_ratio=res.sharpe_ratio,
        status=res.status,
        method="target_return",
    )


@router.post("/efficient-frontier", response_model=EfficientFrontierResponse)
def efficient_frontier(
    req: EfficientFrontierRequest,
    db: Session = Depends(get_db),
) -> EfficientFrontierResponse:
    tickers = _resolve_tickers(req.tickers, req.filters, db)
    price_df = get_price_data(tickers, req.lookback_days, db)
    svc = OptimizeService(lookback_days=req.lookback_days, risk_free_rate=req.risk_free_rate)
    try:
        ef_result = svc.efficient_frontier(
            price_df, n_points=req.n_points, weight_bounds=(0.0, req.max_weight_per_stock)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return EfficientFrontierResponse(
        frontier_points=[FrontierPoint(**p) for p in ef_result.points],
        max_sharpe_point=ef_result.max_sharpe_point,
        min_vol_point=ef_result.min_vol_point,
        tickers=ef_result.tickers,
    )
