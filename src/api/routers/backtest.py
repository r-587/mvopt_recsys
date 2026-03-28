from __future__ import annotations

import logging

import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.deps import get_db, get_price_data, get_universe_tickers
from src.api.schemas.request import BacktestRequest
from src.api.schemas.response import BacktestResponse
from src.services.backtest import BacktestService

router = APIRouter(prefix="/backtest", tags=["backtest"])
logger = logging.getLogger(__name__)


@router.post("/run", response_model=BacktestResponse)
def run_backtest(
    req: BacktestRequest,
    db: Session = Depends(get_db),
) -> BacktestResponse:
    if req.start_date >= req.end_date:
        raise HTTPException(status_code=422, detail="start_date は end_date より前にしてください")

    tickers = req.tickers or get_universe_tickers(
        sectors=req.filters.sectors,
        markets=req.filters.markets,
        max_stocks=req.filters.max_stocks,
        db=db,
    )

    # バックテスト全期間の価格データ（lookback を含めて余裕をもって取得）
    from datetime import timedelta
    extended_lookback = req.lookback_days + (req.end_date - req.start_date).days
    price_df = get_price_data(tickers, extended_lookback, db)

    # ベンチマーク（TOPIX）
    bm_prices = None
    if req.include_benchmark:
        try:
            import pandas as pd
            bm_raw = yf.download(
                "^TOPX",
                start=req.start_date.isoformat(),
                end=req.end_date.isoformat(),
                progress=False,
                auto_adjust=False,
            )
            if not bm_raw.empty:
                bm_close = bm_raw["Adj Close"] if "Adj Close" in bm_raw.columns else bm_raw["Close"]
                if hasattr(bm_close, "squeeze"):
                    bm_close = bm_close.squeeze()
                bm_close.index = bm_close.index.date
                bm_prices = bm_close
        except Exception as e:
            logger.warning("ベンチマーク取得失敗: %s", e)

    svc = BacktestService()
    try:
        result = svc.run(
            price_df=price_df,
            start_date=req.start_date,
            end_date=req.end_date,
            rebalance_freq=req.rebalance_freq,
            lookback_days=req.lookback_days,
            transaction_cost=req.transaction_cost,
            max_weight_per_stock=req.max_weight_per_stock,
            benchmark_prices=bm_prices,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("バックテストエラー")
        raise HTTPException(status_code=500, detail=str(e))

    return BacktestResponse(
        portfolio_values={str(k): v for k, v in result["portfolio_values"].items()},
        benchmark_values={str(k): v for k, v in result["benchmark_values"].items()} if result["benchmark_values"] else None,
        rebalance_dates=result["rebalance_dates"],
        metrics=result["metrics"],
        tickers=result["tickers"],
    )
