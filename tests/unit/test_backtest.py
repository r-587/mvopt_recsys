from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, _get_rebalance_dates
from src.backtest.evaluator import BacktestEvaluator


class TestGetRebalanceDates:
    def test_monthly(self):
        dates = pd.bdate_range("2023-01-01", "2023-12-31").date.tolist()
        rdates = _get_rebalance_dates(dates, "monthly")
        assert len(rdates) == 12  # 各月 1 回

    def test_quarterly(self):
        dates = pd.bdate_range("2023-01-01", "2023-12-31").date.tolist()
        rdates = _get_rebalance_dates(dates, "quarterly")
        assert len(rdates) == 4

    def test_annually(self):
        dates = pd.bdate_range("2021-01-01", "2023-12-31").date.tolist()
        rdates = _get_rebalance_dates(dates, "annually")
        assert len(rdates) == 3


class TestBacktestEngine:
    def test_run_returns_portfolio_values(self, synthetic_prices):
        engine = BacktestEngine()
        prices = synthetic_prices.copy()
        prices.index = prices.index.date

        start = prices.index[252]
        end = prices.index[-1]

        result = engine.run(
            price_data=prices,
            start_date=start,
            end_date=end,
            rebalance_freq="quarterly",
            lookback_days=120,
        )
        assert len(result.portfolio_values) > 0
        assert result.portfolio_values.iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_run_portfolio_values_always_positive(self, synthetic_prices):
        engine = BacktestEngine()
        prices = synthetic_prices.copy()
        prices.index = prices.index.date

        start = prices.index[252]
        end = prices.index[-1]

        result = engine.run(prices, start, end, rebalance_freq="quarterly", lookback_days=120)
        assert (result.portfolio_values > 0).all()

    def test_run_weights_history_not_empty(self, synthetic_prices):
        engine = BacktestEngine()
        prices = synthetic_prices.copy()
        prices.index = prices.index.date

        start = prices.index[252]
        end = prices.index[-1]

        result = engine.run(prices, start, end, rebalance_freq="quarterly", lookback_days=120)
        assert not result.weights_history.empty


class TestBacktestEvaluator:
    def _make_values(self) -> pd.Series:
        n = 252
        rng = np.random.default_rng(0)
        r = rng.normal(0.0003, 0.01, n)
        return pd.Series(np.cumprod(1 + r))

    def test_evaluate_returns_expected_keys(self):
        from src.backtest.engine import BacktestResult
        pv = self._make_values()
        result = BacktestResult(
            portfolio_values=pv,
            weights_history=pd.DataFrame(),
            rebalance_dates=[],
            tickers=[],
        )
        evaluator = BacktestEvaluator()
        metrics = evaluator.evaluate(result)
        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    def test_max_drawdown_negative(self):
        from src.backtest.engine import BacktestResult
        pv = pd.Series([1.0, 1.1, 0.9, 1.0, 0.85, 1.2])
        result = BacktestResult(portfolio_values=pv, weights_history=pd.DataFrame(), rebalance_dates=[], tickers=[])
        evaluator = BacktestEvaluator()
        metrics = evaluator.evaluate(result)
        assert metrics["max_drawdown"] < 0

    def test_information_ratio_with_benchmark(self):
        from src.backtest.engine import BacktestResult
        rng = np.random.default_rng(1)
        pv = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.01, 252)))
        bm = pd.Series(np.cumprod(1 + rng.normal(0.0003, 0.008, 252)))
        result = BacktestResult(portfolio_values=pv, weights_history=pd.DataFrame(), rebalance_dates=[], tickers=[])
        evaluator = BacktestEvaluator()
        metrics = evaluator.evaluate(result, benchmark_values=bm)
        assert "information_ratio" in metrics
        assert "alpha" in metrics
