from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.optimization.base import OptimizationResult
from src.optimization.mean_variance import MeanVarianceOptimizer

logger = logging.getLogger(__name__)


@dataclass
class EfficientFrontierResult:
    points: list[dict]           # [{expected_return, volatility, sharpe_ratio, weights}]
    max_sharpe_point: dict
    min_vol_point: dict
    tickers: list[str] = field(default_factory=list)


class EfficientFrontier:
    """効率的フロンティアの計算"""

    def compute(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        n_points: int = 50,
        risk_free_rate: float = 0.001,
        optimizer: MeanVarianceOptimizer | None = None,
    ) -> EfficientFrontierResult:
        opt = optimizer or MeanVarianceOptimizer()

        # 最小分散・最大シャープレシオを計算
        min_vol_res = opt.min_volatility(expected_returns, cov_matrix)
        max_sharpe_res = opt.max_sharpe(expected_returns, cov_matrix, risk_free_rate)

        # リターン範囲を決定
        r_min = min_vol_res.expected_return
        r_max = float(expected_returns.max())
        if r_max <= r_min:
            r_max = r_min * 1.5 + 0.01

        targets = np.linspace(r_min, r_max, n_points)
        points = []

        for target in targets:
            try:
                res = opt.target_return(expected_returns, cov_matrix, target, risk_free_rate)
                if res.status in ("optimal", "optimal_inaccurate", "fallback"):
                    points.append(
                        {
                            "expected_return": res.expected_return,
                            "volatility": res.volatility,
                            "sharpe_ratio": res.sharpe_ratio,
                            "weights": res.weights_dict,
                        }
                    )
            except Exception as e:
                logger.debug("フロンティア計算スキップ (target=%.4f): %s", target, e)

        if not points:
            points = [
                {
                    "expected_return": min_vol_res.expected_return,
                    "volatility": min_vol_res.volatility,
                    "sharpe_ratio": min_vol_res.sharpe_ratio,
                    "weights": min_vol_res.weights_dict,
                }
            ]

        return EfficientFrontierResult(
            points=points,
            max_sharpe_point={
                "expected_return": max_sharpe_res.expected_return,
                "volatility": max_sharpe_res.volatility,
                "sharpe_ratio": max_sharpe_res.sharpe_ratio,
            },
            min_vol_point={
                "expected_return": min_vol_res.expected_return,
                "volatility": min_vol_res.volatility,
                "sharpe_ratio": min_vol_res.sharpe_ratio,
            },
            tickers=expected_returns.index.tolist(),
        )
