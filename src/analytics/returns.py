from __future__ import annotations

import numpy as np
import pandas as pd


ANNUALIZATION_FACTOR = 252  # 年間営業日数


class ReturnsCalculator:
    """期待リターン推定クラス"""

    def __init__(self, annualization_factor: int = ANNUALIZATION_FACTOR) -> None:
        self._freq = annualization_factor

    def daily_returns(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """日次対数リターン"""
        return np.log(price_df / price_df.shift(1)).dropna(how="all")

    def mean_historical_return(
        self,
        price_df: pd.DataFrame,
        use_log: bool = True,
    ) -> pd.Series:
        """過去リターンの算術平均（年率換算）"""
        r = self.daily_returns(price_df) if use_log else price_df.pct_change().dropna(how="all")
        return r.mean() * self._freq

    def geometric_mean_return(self, price_df: pd.DataFrame) -> pd.Series:
        """幾何平均リターン（年率換算）"""
        r = price_df.pct_change().dropna(how="all")
        return (1 + r).prod() ** (self._freq / len(r)) - 1

    def capm_return(
        self,
        price_df: pd.DataFrame,
        market_returns: pd.Series,
        risk_free_rate: float = 0.001,
        market_premium: float = 0.05,
    ) -> pd.Series:
        """CAPM ベースの期待リターン推定: r_f + β × (E[r_m] - r_f)"""
        stock_r = self.daily_returns(price_df)
        market_r = np.log(market_returns / market_returns.shift(1)).dropna()

        common_idx = stock_r.index.intersection(market_r.index)
        stock_r = stock_r.loc[common_idx]
        mkt = market_r.loc[common_idx]

        betas = {}
        for ticker in stock_r.columns:
            s = stock_r[ticker].dropna()
            m = mkt.loc[s.index]
            if len(s) < 30:
                betas[ticker] = 1.0
                continue
            cov = np.cov(s, m)
            betas[ticker] = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0

        beta_series = pd.Series(betas)
        return risk_free_rate + beta_series * market_premium

    def james_stein_return(
        self,
        price_df: pd.DataFrame,
        shrinkage_target: float = 0.0,
    ) -> pd.Series:
        """James-Stein 収縮推定量"""
        mu = self.mean_historical_return(price_df)
        n = len(mu)
        daily_r = self.daily_returns(price_df)
        sigma2 = daily_r.var().mean() * self._freq

        # 収縮係数
        k = (n - 2) * sigma2 / ((mu - shrinkage_target) ** 2).sum()
        k = min(k, 1.0)
        return shrinkage_target + (1 - k) * (mu - shrinkage_target)
