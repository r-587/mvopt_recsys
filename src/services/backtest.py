from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult, RebalanceFreq
from src.backtest.evaluator import BacktestEvaluator
from src.config import settings
from src.optimization.mean_variance import MeanVarianceOptimizer


class BacktestService:
    """バックテストサービス"""

    def __init__(
        self,
        covariance_method: str = settings.default_covariance_method,
        risk_free_rate: float = settings.default_risk_free_rate,
    ) -> None:
        self._engine = BacktestEngine(covariance_method=covariance_method)
        self._evaluator = BacktestEvaluator()
        self._rf = risk_free_rate

    def run(
        self,
        price_df: pd.DataFrame,
        start_date: date,
        end_date: date,
        rebalance_freq: RebalanceFreq = "monthly",
        lookback_days: int = settings.default_lookback_days,
        transaction_cost: float = 0.001,
        max_weight_per_stock: float = 0.3,
        benchmark_prices: pd.Series | None = None,
    ) -> dict:
        opt = MeanVarianceOptimizer(weight_bounds=(0.0, max_weight_per_stock))

        result = self._engine.run(
            price_data=price_df,
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq,
            optimizer=opt,
            lookback_days=lookback_days,
            transaction_cost=transaction_cost,
            risk_free_rate=self._rf,
        )

        # ベンチマーク価値系列を生成
        bm_values = None
        if benchmark_prices is not None:
            bm_aligned = benchmark_prices.reindex(result.portfolio_values.index).ffill()
            if not bm_aligned.empty:
                bm_values = bm_aligned / bm_aligned.iloc[0]

        metrics = self._evaluator.evaluate(result, bm_values, self._rf)

        return {
            "portfolio_values": result.portfolio_values.to_dict(),
            "benchmark_values": bm_values.to_dict() if bm_values is not None else None,
            "weights_history": result.weights_history.to_dict(),
            "rebalance_dates": [str(d) for d in result.rebalance_dates],
            "metrics": metrics,
            "tickers": result.tickers,
        }
