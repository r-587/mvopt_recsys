from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

from src.optimization.base import OptimizationResult

logger = logging.getLogger(__name__)

_SOLVER_OPTS = {
    "solver": cp.CLARABEL,
    "verbose": False,
}


class MeanVarianceOptimizer:
    """
    Markowitz の平均分散最適化。

    - max_sharpe    : シャープレシオ最大化（SOCP 変換）
    - min_volatility: 最小分散
    - target_return : 目標リターン・リスク最小化
    """

    def __init__(
        self,
        allow_short: bool = False,
        weight_bounds: tuple[float, float] = (0.0, 1.0),
        sector_constraints: dict[str, float] | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self._allow_short = allow_short
        self._lb, self._ub = weight_bounds
        self._sector_constraints = sector_constraints or {}
        self._sector_map = sector_map or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def max_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.001,
    ) -> OptimizationResult:
        """シャープレシオ最大化（Charnes-Cooper SOCP 変換）

        変換: y = t * w, kappa = t = 1 / (excess' w)
        - excess' y = 1  （スケール固定、unbounded を防ぐ）
        - sum(y) = kappa （予算制約: ones'w = 1 に対応）
        - min kappa_vol s.t. ||L y|| <= kappa_vol  （分散最小化 ↔ シャープ最大化）
        """
        mu = expected_returns.values
        Sigma = cov_matrix.values
        n = len(mu)
        excess = mu - risk_free_rate

        if np.all(excess <= 0):
            logger.warning("全銘柄の超過リターンが 0 以下。最小分散にフォールバック")
            return self.min_volatility(expected_returns, cov_matrix)

        L = np.linalg.cholesky(Sigma + np.eye(n) * 1e-8)

        # ||Ly|| <= 1 でリスクスケールを固定 → 有界な凸集合上で excess'y を最大化
        # w* = y* / sum(y*) で実際のウェイトに変換
        y = cp.Variable(n, nonneg=True)
        kappa = cp.Variable(nonneg=True)  # = sum(y) = 予算正規化因子

        objective = cp.Maximize(excess @ y)
        constraints = [
            cp.norm(L @ y, 2) <= 1,    # リスクスケール固定（可行集合を有界にする）
            cp.sum(y) == kappa,
            kappa >= 1e-8,             # 退化解を防ぐ
        ]
        constraints += self._weight_bound_constraints_socp(y, kappa, n)
        constraints += self._sector_constraints_socp(y, kappa, expected_returns.index.tolist())

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(**_SOLVER_OPTS)
        except cp.SolverError as e:
            logger.error("max_sharpe ソルバーエラー: %s", e)
            return self.min_volatility(expected_returns, cov_matrix)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or kappa.value is None or kappa.value < 1e-10:
            logger.warning("max_sharpe 求解失敗 (status=%s)。最小分散にフォールバック", prob.status)
            return self.min_volatility(expected_returns, cov_matrix)

        w = y.value / kappa.value
        w = self._clean_weights(w)
        return self._build_result(w, expected_returns, cov_matrix, risk_free_rate, prob.status)

    def min_volatility(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
    ) -> OptimizationResult:
        """最小分散ポートフォリオ"""
        Sigma = cov_matrix.values
        n = len(expected_returns)
        w = cp.Variable(n)

        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
        constraints = [cp.sum(w) == 1]
        if not self._allow_short:
            constraints.append(w >= self._lb)
        constraints.append(w <= self._ub)
        constraints += self._sector_constraints_regular(w, expected_returns.index.tolist())

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(**_SOLVER_OPTS)
        except cp.SolverError as e:
            logger.error("min_volatility ソルバーエラー: %s", e)
            return self._equal_weight_fallback(expected_returns, cov_matrix)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
            return self._equal_weight_fallback(expected_returns, cov_matrix)

        weights = self._clean_weights(w.value)
        return self._build_result(weights, expected_returns, cov_matrix, status=prob.status)

    def target_return(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target: float,
        risk_free_rate: float = 0.001,
    ) -> OptimizationResult:
        """目標リターンを達成しながら分散を最小化"""
        mu = expected_returns.values
        Sigma = cov_matrix.values
        n = len(mu)
        w = cp.Variable(n)

        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
        constraints = [
            cp.sum(w) == 1,
            mu @ w >= target,
        ]
        if not self._allow_short:
            constraints.append(w >= self._lb)
        constraints.append(w <= self._ub)
        constraints += self._sector_constraints_regular(w, expected_returns.index.tolist())

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(**_SOLVER_OPTS)
        except cp.SolverError as e:
            logger.error("target_return ソルバーエラー: %s", e)
            return self.max_sharpe(expected_returns, cov_matrix, risk_free_rate)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
            logger.warning("target_return 実行不可能 (target=%.4f)。max_sharpe にフォールバック", target)
            return self.max_sharpe(expected_returns, cov_matrix, risk_free_rate)

        weights = self._clean_weights(w.value)
        return self._build_result(weights, expected_returns, cov_matrix, risk_free_rate, prob.status)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _weight_bound_constraints_socp(
        self, y: cp.Variable, kappa: cp.Variable, n: int
    ) -> list:
        constraints = []
        if self._lb > 0:
            constraints.append(y >= self._lb * kappa)
        if self._ub < 1.0:
            constraints.append(y <= self._ub * kappa)
        return constraints

    def _sector_constraints_socp(
        self, y: cp.Variable, kappa: cp.Variable, tickers: list[str]
    ) -> list:
        constraints = []
        for sector, max_w in self._sector_constraints.items():
            idx = [i for i, t in enumerate(tickers) if self._sector_map.get(t) == sector]
            if idx:
                constraints.append(cp.sum(y[idx]) <= max_w * kappa)
        return constraints

    def _sector_constraints_regular(
        self, w: cp.Variable, tickers: list[str]
    ) -> list:
        constraints = []
        for sector, max_w in self._sector_constraints.items():
            idx = [i for i, t in enumerate(tickers) if self._sector_map.get(t) == sector]
            if idx:
                constraints.append(cp.sum(w[idx]) <= max_w)
        return constraints

    @staticmethod
    def _clean_weights(w: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
        w = np.clip(w, 0, None)
        w[w < threshold] = 0.0
        total = w.sum()
        return w / total if total > 0 else np.ones(len(w)) / len(w)

    def _build_result(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.001,
        status: str = "optimal",
    ) -> OptimizationResult:
        from src.analytics.risk_metrics import RiskMetrics
        ret = RiskMetrics.annualized_return(weights, expected_returns)
        vol = RiskMetrics.annualized_volatility(weights, cov_matrix)
        sr = (ret - risk_free_rate) / vol if vol > 0 else 0.0
        return OptimizationResult(
            weights=pd.Series(weights, index=expected_returns.index),
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sr,
            status=str(status),
        )

    def _equal_weight_fallback(
        self, expected_returns: pd.Series, cov_matrix: pd.DataFrame
    ) -> OptimizationResult:
        n = len(expected_returns)
        w = np.ones(n) / n
        return self._build_result(w, expected_returns, cov_matrix, status="fallback")
