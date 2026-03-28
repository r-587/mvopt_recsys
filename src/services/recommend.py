from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import pandas as pd

from src.analytics.risk_metrics import RiskMetrics
from src.config import settings
from src.data.processors.price import PriceProcessor
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.analytics.covariance import get_covariance_estimator
from src.analytics.returns import ReturnsCalculator

logger = logging.getLogger(__name__)

RiskTolerance = Literal["low", "medium", "high"]

_RISK_TARGET_RETURN = {
    "low": None,       # 最小分散
    "medium": None,    # 最大シャープレシオ
    "high": 0.15,      # 目標リターン 15%
}


@dataclass
class PortfolioItem:
    ticker: str
    name: str
    sector: str | None
    weight: float
    shares: int
    amount: int
    expected_return_contribution: float


@dataclass
class RecommendationResult:
    portfolio: list[PortfolioItem]
    metrics: dict[str, float]
    total_invested: int
    cash_remaining: int
    computed_at: date
    optimization_method: str = "max_sharpe"
    warning: str | None = None


class RecommendService:
    """推薦サービス：ユーザー入力からポートフォリオを推薦する"""

    def __init__(
        self,
        lookback_days: int = settings.default_lookback_days,
        covariance_method: str = settings.default_covariance_method,
        risk_free_rate: float = settings.default_risk_free_rate,
    ) -> None:
        self._lookback_days = lookback_days
        self._cov_method = covariance_method
        self._rf = risk_free_rate
        self._calc = ReturnsCalculator()
        self._processor = PriceProcessor()

    def recommend(
        self,
        price_df: pd.DataFrame,
        stock_info: pd.DataFrame,             # index=ticker, columns=[name, sector_33, lot_size, ...]
        risk_tolerance: RiskTolerance = "medium",
        investment_amount: int = 1_000_000,
        max_weight_per_stock: float = 0.3,
        max_weight_per_sector: float = 0.5,
    ) -> RecommendationResult:
        # 前処理
        clean = self._processor.clean(price_df)
        if clean.shape[1] < 2:
            raise ValueError("有効な銘柄が不足しています（最低 2 銘柄必要）")

        returns = self._calc.daily_returns(clean)
        mu = self._calc.mean_historical_return(clean)
        cov_est = get_covariance_estimator(self._cov_method)
        cov = cov_est.fit(returns)

        # 業種マップ
        sector_map: dict[str, str] = {}
        if not stock_info.empty and "sector_33" in stock_info.columns:
            sector_map = stock_info["sector_33"].dropna().to_dict()

        # 業種制約
        sector_constraints: dict[str, float] = {}
        if max_weight_per_sector < 1.0:
            for sector in set(sector_map.values()):
                sector_constraints[sector] = max_weight_per_sector

        opt = MeanVarianceOptimizer(
            weight_bounds=(0.0, max_weight_per_stock),
            sector_constraints=sector_constraints,
            sector_map=sector_map,
        )

        # リスク許容度に応じた最適化
        target = _RISK_TARGET_RETURN[risk_tolerance]
        if risk_tolerance == "low":
            result = opt.min_volatility(mu, cov)
            method = "min_volatility"
        elif risk_tolerance == "high" and target is not None:
            result = opt.target_return(mu, cov, target, self._rf)
            method = "target_return"
        else:
            result = opt.max_sharpe(mu, cov, self._rf)
            method = "max_sharpe"

        # 投資金額に応じた株数計算
        items, total_invested = self._compute_allocation(
            result.weights, stock_info, investment_amount, mu
        )

        # リスク指標
        w_arr = result.weights.values
        metrics = RiskMetrics.portfolio_summary(w_arr, mu, cov, returns, self._rf)

        return RecommendationResult(
            portfolio=items,
            metrics=metrics,
            total_invested=total_invested,
            cash_remaining=investment_amount - total_invested,
            computed_at=date.today(),
            optimization_method=method,
            warning=None if result.status in ("optimal", "optimal_inaccurate") else f"最適化ステータス: {result.status}",
        )

    def _compute_allocation(
        self,
        weights: pd.Series,
        stock_info: pd.DataFrame,
        investment_amount: int,
        mu: pd.Series,
    ) -> tuple[list[PortfolioItem], int]:
        items: list[PortfolioItem] = []
        total_invested = 0

        # ウェイトが正の銘柄のみ
        active = weights[weights > 0]

        for ticker, w in active.items():
            if stock_info.empty or ticker not in stock_info.index:
                name = ticker
                sector = None
                lot_size = 100
            else:
                info = stock_info.loc[ticker]
                name = info.get("name", ticker)
                sector = info.get("sector_33")
                lot_size = int(info.get("lot_size", 100))

            alloc_amount = int(investment_amount * w)
            # 簡易価格推定（最新終値は stock_info に含まれていることを前提）
            latest_price = stock_info.loc[ticker, "latest_price"] if (
                not stock_info.empty
                and ticker in stock_info.index
                and "latest_price" in stock_info.columns
            ) else None

            if latest_price and latest_price > 0:
                price_per_lot = float(latest_price) * lot_size
                lots = max(1, math.floor(alloc_amount / price_per_lot))
                shares = lots * lot_size
                amount = int(shares * float(latest_price))
            else:
                shares = lot_size
                amount = alloc_amount

            total_invested += amount
            ret_contrib = float(w * mu.get(ticker, 0))

            items.append(
                PortfolioItem(
                    ticker=ticker,
                    name=str(name),
                    sector=str(sector) if sector else None,
                    weight=float(w),
                    shares=shares,
                    amount=amount,
                    expected_return_contribution=ret_contrib,
                )
            )

        items.sort(key=lambda x: x.weight, reverse=True)
        return items, min(total_invested, investment_amount)
