from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.deps import get_db, get_price_data, get_universe_tickers
from src.api.schemas.response import CorrelationResponse, UniverseResponse, StockInfo
from src.data.repository import StockRepository
from src.services.analysis import AnalysisService

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/correlation")
def get_correlation(
    tickers: str = "",
    lookback_days: int = 252,
    db: Session = Depends(get_db),
) -> CorrelationResponse:
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()] if tickers else []
    if not ticker_list:
        ticker_list = get_universe_tickers(None, None, 30, db)

    price_df = get_price_data(ticker_list, lookback_days, db)
    svc = AnalysisService()
    try:
        corr = svc.correlation_matrix(price_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return CorrelationResponse(
        matrix={col: row.to_dict() for col, row in corr.iterrows()},
        tickers=corr.columns.tolist(),
    )


@router.get("/universe", response_model=UniverseResponse)
def get_universe(
    market: str = "",
    sector: str = "",
    db: Session = Depends(get_db),
) -> UniverseResponse:
    try:
        repo = StockRepository(db)
        markets = [m.strip() for m in market.split(",") if m.strip()] or None
        sectors = [s.strip() for s in sector.split(",") if s.strip()] or None
        stocks = repo.get_universe(markets=markets, sectors=sectors)
        items = [
            StockInfo(
                ticker=s.ticker,
                name=s.name,
                sector_33=s.sector_33,
                market=s.market,
                lot_size=s.lot_size,
            )
            for s in stocks
        ]
    except Exception:
        items = []

    return UniverseResponse(stocks=items, total=len(items))
