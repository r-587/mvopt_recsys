from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any


class BasePriceFetcher(ABC):
    """株価データフェッチャーの抽象基底クラス"""

    @abstractmethod
    def fetch_prices(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        """
        Returns:
            list of dicts with keys: ticker, date, open, high, low, close, adj_close, volume
        """

    @abstractmethod
    def fetch_stock_list(self) -> list[dict[str, Any]]:
        """
        Returns:
            list of dicts with keys: ticker, name, sector_33, sector_17, market, lot_size
        """
