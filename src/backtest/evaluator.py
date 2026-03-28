from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.risk_metrics import RiskMetrics
from src.backtest.engine import BacktestResult

ANNUALIZATION_FACTOR = 252


class BacktestEvaluator:
    """バックテスト結果の評価指標計算"""

    def evaluate(
        self,
        result: BacktestResult,
        benchmark_values: pd.Series | None = None,
        risk_free_rate: float = 0.001,
    ) -> dict[str, float]:
        pv = result.portfolio_values
        daily_returns = pv.pct_change(fill_method=None).dropna()

        ann_return = self._annualized_return(pv)
        ann_vol = float(daily_returns.std() * np.sqrt(ANNUALIZATION_FACTOR))
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
        mdd = RiskMetrics.max_drawdown(pv)
        calmar = ann_return / abs(mdd) if mdd != 0 else 0.0
        total_return = float(pv.iloc[-1] / pv.iloc[0] - 1)

        metrics: dict[str, float] = {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "calmar_ratio": calmar,
        }

        if len(daily_returns) >= 20:
            var95 = float(np.percentile(daily_returns, 5))
            cvar95 = float(daily_returns[daily_returns <= var95].mean())
            metrics["var_95_daily"] = var95
            metrics["cvar_95_daily"] = cvar95

        if benchmark_values is not None:
            bm = benchmark_values.reindex(pv.index).ffill()
            bm_returns = bm.pct_change(fill_method=None).dropna()
            common = daily_returns.index.intersection(bm_returns.index)
            if len(common) > 10:
                pr = daily_returns.loc[common]
                br = bm_returns.loc[common]
                bm_ann = self._annualized_return(bm)
                excess = pr - br
                te = float(excess.std() * np.sqrt(ANNUALIZATION_FACTOR))
                metrics["information_ratio"] = (
                    float(excess.mean() * ANNUALIZATION_FACTOR) / te if te > 0 else 0.0
                )
                metrics["benchmark_annualized_return"] = bm_ann
                metrics["alpha"] = ann_return - bm_ann

        return metrics

    @staticmethod
    def _annualized_return(values: pd.Series) -> float:
        if len(values) < 2:
            return 0.0
        total_ret = values.iloc[-1] / values.iloc[0] - 1
        n_years = len(values) / ANNUALIZATION_FACTOR
        if n_years <= 0:
            return float(total_ret)
        return float((1 + total_ret) ** (1 / n_years) - 1)

    def summary_table(
        self,
        result: BacktestResult,
        benchmark_values: pd.Series | None = None,
        risk_free_rate: float = 0.001,
    ) -> pd.DataFrame:
        metrics = self.evaluate(result, benchmark_values, risk_free_rate)
        fmt = {
            "total_return": "{:.2%}",
            "annualized_return": "{:.2%}",
            "annualized_volatility": "{:.2%}",
            "sharpe_ratio": "{:.3f}",
            "max_drawdown": "{:.2%}",
            "calmar_ratio": "{:.3f}",
            "var_95_daily": "{:.4f}",
            "cvar_95_daily": "{:.4f}",
            "information_ratio": "{:.3f}",
            "benchmark_annualized_return": "{:.2%}",
            "alpha": "{:.2%}",
        }
        rows = []
        labels = {
            "total_return": "総リターン",
            "annualized_return": "年率リターン",
            "annualized_volatility": "年率ボラティリティ",
            "sharpe_ratio": "シャープレシオ",
            "max_drawdown": "最大ドローダウン",
            "calmar_ratio": "カルマー比",
            "var_95_daily": "日次 VaR(95%)",
            "cvar_95_daily": "日次 CVaR(95%)",
            "information_ratio": "情報比率",
            "benchmark_annualized_return": "ベンチマーク年率",
            "alpha": "アルファ",
        }
        for key, val in metrics.items():
            label = labels.get(key, key)
            display = fmt.get(key, "{:.4f}").format(val)
            rows.append({"指標": label, "値": display})
        return pd.DataFrame(rows)
