from __future__ import annotations

import numpy as np
import pytest

from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.risk_parity import RiskParityOptimizer
from src.optimization.efficient_frontier import EfficientFrontier


class TestMeanVarianceOptimizer:
    def test_max_sharpe_weights_sum_to_one(self, expected_returns, cov_matrix):
        opt = MeanVarianceOptimizer()
        res = opt.max_sharpe(expected_returns, cov_matrix, risk_free_rate=0.001)
        assert abs(res.weights.sum() - 1.0) < 1e-6

    def test_max_sharpe_weights_nonneg(self, expected_returns, cov_matrix):
        opt = MeanVarianceOptimizer(allow_short=False)
        res = opt.max_sharpe(expected_returns, cov_matrix)
        assert (res.weights >= -1e-8).all()

    def test_max_sharpe_weight_bound_respected(self, expected_returns, cov_matrix):
        max_w = 0.2
        opt = MeanVarianceOptimizer(weight_bounds=(0.0, max_w))
        res = opt.max_sharpe(expected_returns, cov_matrix)
        assert (res.weights <= max_w + 1e-6).all()

    def test_min_volatility_weights_sum_to_one(self, expected_returns, cov_matrix):
        opt = MeanVarianceOptimizer()
        res = opt.min_volatility(expected_returns, cov_matrix)
        assert abs(res.weights.sum() - 1.0) < 1e-6

    def test_min_volatility_lower_risk_than_equal_weight(self, expected_returns, cov_matrix):
        from src.analytics.risk_metrics import RiskMetrics
        n = len(expected_returns)
        eq_w = np.ones(n) / n
        eq_vol = RiskMetrics.annualized_volatility(eq_w, cov_matrix)

        opt = MeanVarianceOptimizer()
        res = opt.min_volatility(expected_returns, cov_matrix)
        # 最小分散は等ウェイトよりリスクが低いはず
        assert res.volatility <= eq_vol + 1e-6

    def test_target_return_achieves_target(self, expected_returns, cov_matrix):
        mu_min = float(expected_returns.min())
        mu_max = float(expected_returns.max())
        target = mu_min + (mu_max - mu_min) * 0.3

        opt = MeanVarianceOptimizer()
        res = opt.target_return(expected_returns, cov_matrix, target)
        # 実現リターンが目標以上（または近似）
        assert res.expected_return >= target - 0.001

    def test_infeasible_target_falls_back(self, expected_returns, cov_matrix):
        """実行不可能な目標リターンは max_sharpe にフォールバック"""
        opt = MeanVarianceOptimizer()
        res = opt.target_return(expected_returns, cov_matrix, target=99.0)
        assert res.status in ("optimal", "optimal_inaccurate", "fallback")
        assert abs(res.weights.sum() - 1.0) < 1e-4

    def test_sector_constraint(self, expected_returns, cov_matrix):
        tickers = expected_returns.index.tolist()
        sector_map = {t: "A" if i < 10 else "B" for i, t in enumerate(tickers)}
        sector_constraints = {"A": 0.4}

        opt = MeanVarianceOptimizer(sector_constraints=sector_constraints, sector_map=sector_map)
        res = opt.max_sharpe(expected_returns, cov_matrix)

        sector_a_weight = sum(
            res.weights.get(t, 0) for t in tickers if sector_map.get(t) == "A"
        )
        assert sector_a_weight <= 0.4 + 1e-4


class TestRiskParityOptimizer:
    def test_weights_sum_to_one(self, expected_returns, cov_matrix):
        opt = RiskParityOptimizer()
        res = opt.optimize(expected_returns, cov_matrix)
        assert abs(res.weights.sum() - 1.0) < 1e-4

    def test_risk_contributions_approximately_equal(self, expected_returns, cov_matrix):
        from src.analytics.risk_metrics import RiskMetrics
        opt = RiskParityOptimizer()
        res = opt.optimize(expected_returns, cov_matrix)
        rc = RiskMetrics.risk_contribution(res.weights.values, cov_matrix)
        # 最大 RC と最小 RC の差が小さいこと
        assert rc.max() - rc.min() < 0.1  # 10% 以内の差


class TestEfficientFrontier:
    def test_frontier_has_multiple_points(self, expected_returns, cov_matrix):
        ef = EfficientFrontier()
        result = ef.compute(expected_returns, cov_matrix, n_points=10)
        assert len(result.points) >= 2

    def test_frontier_monotone_risk_return(self, expected_returns, cov_matrix):
        ef = EfficientFrontier()
        result = ef.compute(expected_returns, cov_matrix, n_points=20)
        vols = [p["volatility"] for p in result.points]
        rets = [p["expected_return"] for p in result.points]
        # フロンティア上ではリスクとリターンが概ね単調
        # 完全単調でなくても大まかな正相関を確認
        corr = np.corrcoef(vols, rets)[0, 1]
        assert corr > 0

    def test_max_sharpe_in_frontier_range(self, expected_returns, cov_matrix):
        ef = EfficientFrontier()
        result = ef.compute(expected_returns, cov_matrix, n_points=20)
        ms = result.max_sharpe_point
        rets = [p["expected_return"] for p in result.points]
        assert min(rets) - 0.01 <= ms["expected_return"] <= max(rets) + 0.01
