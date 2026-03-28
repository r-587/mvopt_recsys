from __future__ import annotations

import pandas as pd

from src.analytics.covariance import get_covariance_estimator
from src.analytics.returns import ReturnsCalculator
from src.analytics.risk_metrics import RiskMetrics
from src.config import settings
from src.data.processors.price import PriceProcessor


class AnalysisService:
    """分析サービス：相関行列・リスク指標・ベンチマーク比較"""

    def __init__(
        self,
        covariance_method: str = settings.default_covariance_method,
    ) -> None:
        self._cov_method = covariance_method
        self._calc = ReturnsCalculator()
        self._processor = PriceProcessor()

    def correlation_matrix(self, price_df: pd.DataFrame) -> pd.DataFrame:
        clean = self._processor.clean(price_df)
        returns = self._calc.daily_returns(clean)
        return returns.corr()

    def covariance_matrix(self, price_df: pd.DataFrame) -> pd.DataFrame:
        clean = self._processor.clean(price_df)
        returns = self._calc.daily_returns(clean)
        cov_est = get_covariance_estimator(self._cov_method)
        return cov_est.fit(returns)

    def expected_returns(self, price_df: pd.DataFrame) -> pd.Series:
        clean = self._processor.clean(price_df)
        return self._calc.mean_historical_return(clean)

    def portfolio_metrics(
        self,
        weights: dict[str, float],
        price_df: pd.DataFrame,
        risk_free_rate: float = settings.default_risk_free_rate,
    ) -> dict[str, float]:
        import numpy as np

        clean = self._processor.clean(price_df)
        returns = self._calc.daily_returns(clean)
        mu = self._calc.mean_historical_return(clean)
        cov_est = get_covariance_estimator(self._cov_method)
        cov = cov_est.fit(returns)

        # ウェイトをベクトルに変換
        common = [t for t in weights if t in mu.index]
        if not common:
            return {}
        w_series = pd.Series(weights).reindex(common).fillna(0)
        w_series /= w_series.sum()

        w_arr = w_series.values
        mu_aligned = mu.reindex(common)
        cov_aligned = cov.reindex(common, axis=0).reindex(common, axis=1)
        returns_aligned = returns.reindex(columns=common).dropna()

        return RiskMetrics.portfolio_summary(w_arr, mu_aligned, cov_aligned, returns_aligned, risk_free_rate)

    def sector_breakdown(
        self,
        weights: dict[str, float],
        sector_map: dict[str, str],
    ) -> dict[str, float]:
        breakdown: dict[str, float] = {}
        for ticker, w in weights.items():
            sector = sector_map.get(ticker, "不明")
            breakdown[sector] = breakdown.get(sector, 0.0) + w
        return breakdown
