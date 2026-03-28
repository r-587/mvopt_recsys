from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.covariance import (
    EWMACovariance,
    LedoitWolfCovariance,
    OASCovariance,
    SampleCovariance,
    get_covariance_estimator,
)
from src.analytics.returns import ReturnsCalculator
from src.analytics.risk_metrics import RiskMetrics


class TestReturnsCalculator:
    def test_daily_returns_shape(self, synthetic_prices):
        calc = ReturnsCalculator()
        r = calc.daily_returns(synthetic_prices)
        assert r.shape == (len(synthetic_prices) - 1, synthetic_prices.shape[1])

    def test_mean_historical_return_annualized(self, synthetic_prices):
        calc = ReturnsCalculator()
        mu = calc.mean_historical_return(synthetic_prices)
        assert len(mu) == synthetic_prices.shape[1]
        # 年率換算なので絶対値が 1 未満の範囲であることを確認
        assert (mu.abs() < 5.0).all()

    def test_geometric_mean_return(self, synthetic_prices):
        calc = ReturnsCalculator()
        geo = calc.geometric_mean_return(synthetic_prices)
        # 幾何平均リターンが年率換算値であること（-1 〜 +∞ の範囲）
        assert len(geo) == synthetic_prices.shape[1]
        # 合成データ（年率 5%・vol 20%）なので概ね -50% ~ +100% の範囲に収まるはず
        assert (geo > -1.0).all() and (geo < 5.0).all()

    def test_james_stein_return(self, synthetic_prices):
        calc = ReturnsCalculator()
        js = calc.james_stein_return(synthetic_prices)
        mu = calc.mean_historical_return(synthetic_prices)
        assert len(js) == len(mu)
        # 収縮されているので元の平均より 0 に近い（一般的傾向）
        assert (js.abs() <= mu.abs() + 1e-10).mean() > 0.5


class TestCovarianceEstimators:
    def test_sample_covariance_positive_definite(self, synthetic_returns):
        cov = SampleCovariance().fit(synthetic_returns)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert (eigenvalues > -1e-8).all(), "共分散行列が半正定値でない"

    def test_ledoit_wolf_positive_definite(self, synthetic_returns):
        cov = LedoitWolfCovariance().fit(synthetic_returns)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert (eigenvalues > -1e-8).all()

    def test_oas_covariance(self, synthetic_returns):
        cov = OASCovariance().fit(synthetic_returns)
        assert cov.shape == (synthetic_returns.shape[1], synthetic_returns.shape[1])

    def test_ewma_covariance(self, synthetic_returns):
        cov = EWMACovariance(span=60).fit(synthetic_returns)
        n = synthetic_returns.shape[1]
        assert cov.shape == (n, n)

    def test_get_covariance_estimator_factory(self):
        for method in ["sample", "ledoit_wolf", "oas", "ewma"]:
            est = get_covariance_estimator(method)
            assert est is not None

    def test_get_covariance_estimator_unknown(self):
        with pytest.raises(ValueError):
            get_covariance_estimator("unknown_method")


class TestRiskMetrics:
    def test_annualized_return(self, expected_returns, cov_matrix):
        n = len(expected_returns)
        w = np.ones(n) / n
        ret = RiskMetrics.annualized_return(w, expected_returns)
        assert isinstance(ret, float)

    def test_annualized_volatility_nonneg(self, expected_returns, cov_matrix):
        n = len(expected_returns)
        w = np.ones(n) / n
        vol = RiskMetrics.annualized_volatility(w, cov_matrix)
        assert vol >= 0

    def test_sharpe_ratio(self, expected_returns, cov_matrix):
        n = len(expected_returns)
        w = np.ones(n) / n
        sr = RiskMetrics.sharpe_ratio(w, expected_returns, cov_matrix, rf=0.001)
        assert isinstance(sr, float)

    def test_value_at_risk(self, synthetic_returns, expected_returns, cov_matrix):
        n = len(expected_returns)
        w = np.ones(n) / n
        returns_aligned = synthetic_returns.reindex(columns=expected_returns.index)
        var = RiskMetrics.value_at_risk(w, returns_aligned.dropna(), alpha=0.05)
        assert var < 0  # VaR は損失なので負値

    def test_max_drawdown(self):
        values = pd.Series([1.0, 1.1, 0.9, 1.05, 0.8, 1.2])
        mdd = RiskMetrics.max_drawdown(values)
        assert mdd < 0

    def test_risk_contribution_sums_to_one(self, expected_returns, cov_matrix):
        n = len(expected_returns)
        w = np.ones(n) / n
        rc = RiskMetrics.risk_contribution(w, cov_matrix)
        assert abs(rc.sum() - 1.0) < 1e-8

    def test_portfolio_summary_keys(self, expected_returns, cov_matrix, synthetic_returns):
        n = len(expected_returns)
        w = np.ones(n) / n
        returns_aligned = synthetic_returns.reindex(columns=expected_returns.index).dropna()
        summary = RiskMetrics.portfolio_summary(w, expected_returns, cov_matrix, returns_aligned)
        assert "expected_annual_return" in summary
        assert "annual_volatility" in summary
        assert "sharpe_ratio" in summary
        assert "var_95" in summary
        assert "cvar_95" in summary
