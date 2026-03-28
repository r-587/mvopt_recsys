from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


ANNUALIZATION_FACTOR = 252


class SampleCovariance:
    """標本共分散行列"""

    def fit(self, returns: pd.DataFrame) -> pd.DataFrame:
        cov = returns.cov() * ANNUALIZATION_FACTOR
        return cov


class LedoitWolfCovariance:
    """Ledoit-Wolf 収縮推定量（推奨デフォルト）"""

    def fit(self, returns: pd.DataFrame) -> pd.DataFrame:
        lw = LedoitWolf()
        lw.fit(returns.dropna())
        cov = pd.DataFrame(
            lw.covariance_ * ANNUALIZATION_FACTOR,
            index=returns.columns,
            columns=returns.columns,
        )
        return cov


class OASCovariance:
    """Oracle Approximating Shrinkage 推定量"""

    def fit(self, returns: pd.DataFrame) -> pd.DataFrame:
        oas = OAS()
        oas.fit(returns.dropna())
        cov = pd.DataFrame(
            oas.covariance_ * ANNUALIZATION_FACTOR,
            index=returns.columns,
            columns=returns.columns,
        )
        return cov


class EWMACovariance:
    """指数加重移動平均共分散（直近データを重視）"""

    def __init__(self, span: int = 60) -> None:
        self._span = span

    def fit(self, returns: pd.DataFrame) -> pd.DataFrame:
        df = returns.dropna()
        alpha = 2 / (self._span + 1)
        weights = np.array([(1 - alpha) ** i for i in range(len(df) - 1, -1, -1)])
        weights /= weights.sum()

        demeaned = (df - df.mean()).values  # numpy array に変換
        cov = demeaned.T @ np.diag(weights) @ demeaned
        cov_df = pd.DataFrame(cov * ANNUALIZATION_FACTOR, index=df.columns, columns=df.columns)
        return cov_df


def get_covariance_estimator(method: str = "ledoit_wolf") -> SampleCovariance | LedoitWolfCovariance | OASCovariance | EWMACovariance:
    _map = {
        "sample": SampleCovariance(),
        "ledoit_wolf": LedoitWolfCovariance(),
        "oas": OASCovariance(),
        "ewma": EWMACovariance(),
    }
    if method not in _map:
        raise ValueError(f"未知の共分散推定手法: {method}. 選択肢: {list(_map)}")
    return _map[method]
