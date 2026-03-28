from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from src.analytics.covariance import get_covariance_estimator
from src.analytics.returns import ReturnsCalculator
from src.optimization.mean_variance import MeanVarianceOptimizer

logger = logging.getLogger(__name__)

RebalanceFreq = Literal["monthly", "quarterly", "annually"]


@dataclass
class BacktestResult:
    portfolio_values: pd.Series           # 日次ポートフォリオ価値（初期=1.0）
    weights_history: pd.DataFrame         # 各リバランス日のウェイト
    rebalance_dates: list[date]
    tickers: list[str]
    metrics: dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """
    ポートフォリオバックテストエンジン。
    各リバランス日に MVO を実行し，次のリバランス日まで保有し続ける。
    """

    def __init__(
        self,
        covariance_method: str = "ledoit_wolf",
        return_method: str = "mean_historical",
    ) -> None:
        self._calc = ReturnsCalculator()
        self._cov_method = covariance_method

    def run(
        self,
        price_data: pd.DataFrame,         # index=date, columns=ticker（調整済終値）
        start_date: date,
        end_date: date,
        rebalance_freq: RebalanceFreq = "monthly",
        optimizer: MeanVarianceOptimizer | None = None,
        lookback_days: int = 252,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.001,
    ) -> BacktestResult:
        opt = optimizer or MeanVarianceOptimizer()
        price_data = price_data.sort_index()

        # バックテスト期間内の価格データ
        bt_prices = price_data.loc[
            (price_data.index >= start_date) & (price_data.index <= end_date)
        ]
        if bt_prices.empty:
            raise ValueError("バックテスト期間内に価格データがありません")

        tickers = bt_prices.columns.tolist()
        rebalance_dates = _get_rebalance_dates(bt_prices.index.tolist(), rebalance_freq)

        # ポートフォリオ追跡変数
        portfolio_value = 1.0
        portfolio_values: dict[date, float] = {}
        weights_history: dict[date, dict[str, float]] = {}
        current_weights = np.ones(len(tickers)) / len(tickers)

        prev_date = None
        current_rebalance_idx = 0

        for dt in bt_prices.index:
            # リバランス判定
            if (
                current_rebalance_idx < len(rebalance_dates)
                and dt >= rebalance_dates[current_rebalance_idx]
            ):
                new_weights = self._optimize_weights(
                    price_data, dt, tickers, opt, lookback_days, risk_free_rate
                )
                if new_weights is not None:
                    # 取引コスト控除
                    turnover = np.abs(new_weights - current_weights).sum() / 2
                    portfolio_value *= 1 - transaction_cost * turnover
                    current_weights = new_weights
                    weights_history[dt] = dict(zip(tickers, current_weights))
                current_rebalance_idx += 1

            # 日次リターン計算
            if prev_date is not None:
                try:
                    prev_prices = bt_prices.loc[prev_date, tickers].values
                    curr_prices = bt_prices.loc[dt, tickers].values
                    valid = (prev_prices > 0) & (curr_prices > 0)
                    if valid.any():
                        daily_ret = np.where(valid, curr_prices / prev_prices - 1, 0.0)
                        port_ret = float(current_weights @ daily_ret)
                        portfolio_value *= 1 + port_ret
                        # ウェイトを自然推移
                        new_w = current_weights * (1 + daily_ret)
                        total = new_w.sum()
                        if total > 0:
                            current_weights = new_w / total
                except (KeyError, IndexError):
                    pass

            portfolio_values[dt] = portfolio_value
            prev_date = dt

        pv_series = pd.Series(portfolio_values)
        result = BacktestResult(
            portfolio_values=pv_series,
            weights_history=pd.DataFrame(weights_history).T,
            rebalance_dates=rebalance_dates,
            tickers=tickers,
        )
        return result

    def _optimize_weights(
        self,
        full_price_data: pd.DataFrame,
        as_of: date,
        tickers: list[str],
        optimizer: MeanVarianceOptimizer,
        lookback_days: int,
        risk_free_rate: float,
    ) -> np.ndarray | None:
        # lookback 期間の価格取得
        past_prices = full_price_data.loc[full_price_data.index < as_of, tickers]
        if len(past_prices) < lookback_days // 2:
            return None
        past_prices = past_prices.tail(lookback_days)

        # 欠損銘柄除外
        past_prices = past_prices.dropna(axis=1, thresh=int(len(past_prices) * 0.9))
        if past_prices.empty or past_prices.shape[1] < 2:
            return None

        valid_tickers = past_prices.columns.tolist()
        returns = self._calc.daily_returns(past_prices)
        mu = self._calc.mean_historical_return(past_prices)
        cov_est = get_covariance_estimator(self._cov_method)
        cov = cov_est.fit(returns)

        try:
            res = optimizer.max_sharpe(mu, cov, risk_free_rate)
        except Exception as e:
            logger.debug("バックテスト最適化失敗 (%s): %s", as_of, e)
            return None

        # full tickers に対してウェイトを再マッピング
        full_weights = np.zeros(len(tickers))
        for i, t in enumerate(tickers):
            if t in res.weights.index:
                full_weights[i] = res.weights[t]
        if full_weights.sum() > 0:
            full_weights /= full_weights.sum()
        else:
            full_weights = np.ones(len(tickers)) / len(tickers)
        return full_weights


def _get_rebalance_dates(dates: list, freq: RebalanceFreq) -> list[date]:
    """リバランス日（各月/四半期/年の最初の営業日）を返す"""
    rebalance = []
    prev_key = None
    for dt in dates:
        if freq == "monthly":
            key = (dt.year, dt.month)
        elif freq == "quarterly":
            key = (dt.year, (dt.month - 1) // 3)
        else:  # annually
            key = (dt.year,)
        if key != prev_key:
            rebalance.append(dt)
            prev_key = key
    return rebalance
