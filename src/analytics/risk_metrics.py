from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION_FACTOR = 252


class RiskMetrics:
    """ポートフォリオのリスク指標計算"""

    @staticmethod
    def annualized_return(
        weights: np.ndarray,
        mu: pd.Series,
        freq: int = ANNUALIZATION_FACTOR,
    ) -> float:
        """年率期待リターン（mu は既に年率換算済みを想定）"""
        w = np.asarray(weights)
        return float(w @ mu.values)

    @staticmethod
    def annualized_volatility(
        weights: np.ndarray,
        cov: pd.DataFrame,
        freq: int = ANNUALIZATION_FACTOR,
    ) -> float:
        """年率ボラティリティ（cov は既に年率換算済みを想定）"""
        w = np.asarray(weights)
        variance = w @ cov.values @ w
        return float(np.sqrt(max(variance, 0.0)))

    @staticmethod
    def sharpe_ratio(
        weights: np.ndarray,
        mu: pd.Series,
        cov: pd.DataFrame,
        rf: float = 0.001,
        freq: int = ANNUALIZATION_FACTOR,
    ) -> float:
        """シャープレシオ"""
        ret = RiskMetrics.annualized_return(weights, mu)
        vol = RiskMetrics.annualized_volatility(weights, cov)
        if vol == 0:
            return 0.0
        return (ret - rf) / vol

    @staticmethod
    def value_at_risk(
        weights: np.ndarray,
        returns_df: pd.DataFrame,
        alpha: float = 0.05,
    ) -> float:
        """Historical VaR（負値：損失を正として）"""
        w = np.asarray(weights)
        port_returns = returns_df.values @ w
        return float(np.percentile(port_returns, alpha * 100))

    @staticmethod
    def conditional_var(
        weights: np.ndarray,
        returns_df: pd.DataFrame,
        alpha: float = 0.05,
    ) -> float:
        """Historical CVaR（Expected Shortfall）"""
        w = np.asarray(weights)
        port_returns = returns_df.values @ w
        var = np.percentile(port_returns, alpha * 100)
        tail = port_returns[port_returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    @staticmethod
    def max_drawdown(portfolio_values: pd.Series) -> float:
        """最大ドローダウン（負値）"""
        rolling_max = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        return float(drawdown.min())

    @staticmethod
    def calmar_ratio(
        annualized_return: float,
        portfolio_values: pd.Series,
    ) -> float:
        """カルマー比 = 年率リターン / |最大ドローダウン|"""
        mdd = abs(RiskMetrics.max_drawdown(portfolio_values))
        return annualized_return / mdd if mdd != 0 else 0.0

    @staticmethod
    def risk_contribution(
        weights: np.ndarray,
        cov: pd.DataFrame,
    ) -> np.ndarray:
        """各銘柄のリスク寄与度（比率）"""
        w = np.asarray(weights)
        port_vol = np.sqrt(w @ cov.values @ w)
        if port_vol == 0:
            return np.zeros_like(w)
        marginal = cov.values @ w
        rc = w * marginal / port_vol
        return rc / rc.sum()

    @staticmethod
    def portfolio_summary(
        weights: np.ndarray | pd.Series,
        mu: pd.Series,
        cov: pd.DataFrame,
        returns_df: pd.DataFrame | None = None,
        rf: float = 0.001,
    ) -> dict[str, float]:
        w = np.asarray(weights)
        result = {
            "expected_annual_return": RiskMetrics.annualized_return(w, mu),
            "annual_volatility": RiskMetrics.annualized_volatility(w, cov),
            "sharpe_ratio": RiskMetrics.sharpe_ratio(w, mu, cov, rf),
        }
        if returns_df is not None:
            result["var_95"] = RiskMetrics.value_at_risk(w, returns_df, alpha=0.05)
            result["cvar_95"] = RiskMetrics.conditional_var(w, returns_df, alpha=0.05)
        return result
