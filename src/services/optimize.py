from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from src.analytics.covariance import get_covariance_estimator
from src.analytics.returns import ReturnsCalculator
from src.config import settings
from src.data.processors.price import PriceProcessor
from src.optimization.efficient_frontier import EfficientFrontier, EfficientFrontierResult
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.base import OptimizationResult
from src.optimization.risk_parity import RiskParityOptimizer

logger = logging.getLogger(__name__)


class OptimizeService:
    """最適化サービス：価格データから最適ポートフォリオを計算する"""

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

    def _prepare(
        self, price_df: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """共通前処理：クレンジング → リターン → μ・Σ計算"""
        clean = self._processor.clean(price_df)
        returns = self._calc.daily_returns(clean)
        mu = self._calc.mean_historical_return(clean)
        cov_est = get_covariance_estimator(self._cov_method)
        cov = cov_est.fit(returns)
        return mu, cov, returns

    def max_sharpe(
        self,
        price_df: pd.DataFrame,
        weight_bounds: tuple[float, float] = (0.0, 1.0),
        sector_constraints: dict[str, float] | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> OptimizationResult:
        mu, cov, _ = self._prepare(price_df)
        opt = MeanVarianceOptimizer(
            weight_bounds=weight_bounds,
            sector_constraints=sector_constraints,
            sector_map=sector_map,
        )
        return opt.max_sharpe(mu, cov, self._rf)

    def min_volatility(
        self,
        price_df: pd.DataFrame,
        weight_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> OptimizationResult:
        mu, cov, _ = self._prepare(price_df)
        opt = MeanVarianceOptimizer(weight_bounds=weight_bounds)
        return opt.min_volatility(mu, cov)

    def target_return(
        self,
        price_df: pd.DataFrame,
        target: float,
        weight_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> OptimizationResult:
        mu, cov, _ = self._prepare(price_df)
        opt = MeanVarianceOptimizer(weight_bounds=weight_bounds)
        return opt.target_return(mu, cov, target, self._rf)

    def risk_parity(self, price_df: pd.DataFrame) -> OptimizationResult:
        mu, cov, _ = self._prepare(price_df)
        opt = RiskParityOptimizer()
        return opt.optimize(mu, cov, self._rf)

    def efficient_frontier(
        self,
        price_df: pd.DataFrame,
        n_points: int = 50,
        weight_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> EfficientFrontierResult:
        mu, cov, _ = self._prepare(price_df)
        inner_opt = MeanVarianceOptimizer(weight_bounds=weight_bounds)
        ef = EfficientFrontier()
        return ef.compute(mu, cov, n_points=n_points, risk_free_rate=self._rf, optimizer=inner_opt)
