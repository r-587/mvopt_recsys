from __future__ import annotations

"""pytest 共通フィクスチャ"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_prices() -> pd.DataFrame:
    """20 銘柄 × 500 日の合成株価データ（再現性あり）"""
    rng = np.random.default_rng(42)
    n_assets = 20
    n_days = 500
    tickers = [f"STOCK{i:02d}.T" for i in range(n_assets)]

    # ランダムな日次リターン（年率 5% リターン、年率 20% ボラティリティ相当）
    daily_mu = 0.05 / 252
    daily_sigma = 0.20 / np.sqrt(252)
    returns = rng.normal(daily_mu, daily_sigma, size=(n_days, n_assets))

    # 累積積でリターンを株価に変換
    prices = np.cumprod(1 + returns, axis=0) * 1000.0
    dates = pd.bdate_range(end="2024-12-31", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture(scope="session")
def synthetic_returns(synthetic_prices) -> pd.DataFrame:
    return np.log(synthetic_prices / synthetic_prices.shift(1)).dropna()


@pytest.fixture(scope="session")
def expected_returns(synthetic_prices) -> pd.Series:
    from src.analytics.returns import ReturnsCalculator
    calc = ReturnsCalculator()
    return calc.mean_historical_return(synthetic_prices)


@pytest.fixture(scope="session")
def cov_matrix(synthetic_returns) -> pd.DataFrame:
    from src.analytics.covariance import LedoitWolfCovariance
    return LedoitWolfCovariance().fit(synthetic_returns)
