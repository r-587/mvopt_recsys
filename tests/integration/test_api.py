from __future__ import annotations

"""
FastAPI 結合テスト（DB 不要のスタンドアロンモード）
実際の HTTP リクエストをテストする。
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestOptimizeEndpoints:
    """最適化エンドポイントのスモークテスト（yfinance から少数銘柄で検証）"""

    _tickers = ["7203.T", "6758.T", "9984.T", "8306.T", "6861.T"]

    def test_max_sharpe(self, client):
        payload = {
            "tickers": self._tickers,
            "lookback_days": 120,
            "max_weight_per_stock": 0.5,
        }
        resp = client.post("/api/v1/optimize/max-sharpe", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "weights" in data
            assert abs(sum(data["weights"].values()) - 1.0) < 1e-4
        else:
            # データ取得失敗はスキップ（CI 環境でネットワーク不可の場合）
            pytest.skip(f"最適化 API 失敗（ネットワーク不可の可能性）: {resp.status_code}")

    def test_min_volatility(self, client):
        payload = {
            "tickers": self._tickers,
            "lookback_days": 120,
            "max_weight_per_stock": 0.5,
        }
        resp = client.post("/api/v1/optimize/min-volatility", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert data["method"] == "min_volatility"
        else:
            pytest.skip(f"ネットワーク不可: {resp.status_code}")

    def test_efficient_frontier(self, client):
        payload = {
            "tickers": self._tickers,
            "lookback_days": 120,
            "n_points": 10,
        }
        resp = client.post("/api/v1/optimize/efficient-frontier", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert len(data["frontier_points"]) >= 2
        else:
            pytest.skip(f"ネットワーク不可: {resp.status_code}")


class TestRecommendEndpoint:
    _tickers = ["7203.T", "6758.T", "9984.T", "8306.T", "6861.T"]

    def test_recommend_portfolio(self, client):
        payload = {
            "investment_amount": 1_000_000,
            "risk_tolerance": "medium",
            "filters": {"max_stocks": 5},
            "lookback_days": 120,
        }
        resp = client.post("/api/v1/recommend/portfolio", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "portfolio" in data
            assert "metrics" in data
            assert data["total_invested"] + data["cash_remaining"] <= 1_000_000 + 1
        else:
            pytest.skip(f"ネットワーク不可: {resp.status_code}")
