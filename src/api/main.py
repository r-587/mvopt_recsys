from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import analysis, backtest, optimize, recommend
from src.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="日本個別株推薦システム API",
    description="平均分散最適化（MVO）に基づく日本株ポートフォリオ推薦",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router, prefix="/api/v1")
app.include_router(optimize.router, prefix="/api/v1")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(backtest.router, prefix="/api/v1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.on_event("startup")
def startup_event() -> None:
    """テーブルが存在しなければ作成（開発用）"""
    try:
        from src.data.database import create_tables
        create_tables()
        logging.getLogger(__name__).info("DB テーブル確認完了")
    except Exception as e:
        logging.getLogger(__name__).warning("DB 接続不可（スタンドアロンモードで動作）: %s", e)
