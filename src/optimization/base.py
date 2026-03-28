from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class OptimizationResult:
    weights: pd.Series                    # index=ticker, values=weight
    expected_return: float
    volatility: float
    sharpe_ratio: float
    status: str = "optimal"               # optimal / infeasible / fallback
    solver_info: dict = field(default_factory=dict)

    @property
    def weights_dict(self) -> dict[str, float]:
        return self.weights.to_dict()
