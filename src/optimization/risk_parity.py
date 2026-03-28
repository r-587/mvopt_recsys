from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.optimization.base import OptimizationResult

logger = logging.getLogger(__name__)


class RiskParityOptimizer:
    """
    リスクパリティ最適化：
    各銘柄のリスク寄与度が均等になるようウェイトを決定する。
    scipy の L-BFGS-B で最適化。
    """

    def optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.001,
    ) -> OptimizationResult:
        Sigma = cov_matrix.values
        n = len(expected_returns)
        target_rc = np.ones(n) / n  # 均等リスク寄与

        def _risk_contribution_diff(w: np.ndarray) -> float:
            w = np.abs(w)
            w /= w.sum()
            port_var = w @ Sigma @ w
            port_vol = np.sqrt(port_var)
            if port_vol < 1e-10:
                return 1.0
            marginal = Sigma @ w
            rc = w * marginal / port_vol
            rc_normalized = rc / rc.sum()
            return float(np.sum((rc_normalized - target_rc) ** 2))

        w0 = np.ones(n) / n
        bounds = [(1e-6, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            _risk_contribution_diff,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning("リスクパリティ最適化失敗: %s", result.message)

        weights = np.abs(result.x)
        weights /= weights.sum()
        weights[weights < 1e-4] = 0.0
        weights /= weights.sum()

        from src.analytics.risk_metrics import RiskMetrics
        ret = RiskMetrics.annualized_return(weights, expected_returns)
        vol = RiskMetrics.annualized_volatility(weights, cov_matrix)
        sr = (ret - risk_free_rate) / vol if vol > 0 else 0.0

        return OptimizationResult(
            weights=pd.Series(weights, index=expected_returns.index),
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sr,
            status="optimal" if result.success else "suboptimal",
        )
