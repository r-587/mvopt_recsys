from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

from src.data.fetchers.base import BasePriceFetcher

logger = logging.getLogger(__name__)

# 東証主要銘柄のデフォルトリスト（yfinance 用サフィックス .T）
# J-Quants が利用できない場合のフォールバック
_DEFAULT_JP_TICKERS = [
    "7203.T", "6758.T", "9984.T", "8306.T", "6861.T",
    "9432.T", "7974.T", "6098.T", "8035.T", "4063.T",
    "9433.T", "7751.T", "6954.T", "4502.T", "8058.T",
    "2914.T", "8316.T", "9022.T", "6501.T", "4519.T",
    "9020.T", "6367.T", "8001.T", "4661.T", "3382.T",
    "8031.T", "6702.T", "5108.T", "6645.T", "7267.T",
    "8411.T", "6902.T", "7201.T", "4307.T", "6503.T",
    "8802.T", "2802.T", "6724.T", "4452.T", "7733.T",
    "7011.T", "8604.T", "6762.T", "1925.T", "4689.T",
    "9613.T", "8309.T", "8766.T", "4901.T", "2503.T",
]


class YFinanceFetcher(BasePriceFetcher):
    """yfinance を使った株価データ取得"""

    def fetch_prices(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        batch_size = 50

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            try:
                raw = yf.download(
                    batch,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
            except Exception as e:
                logger.warning("yfinance download error for batch %s: %s", batch, e)
                continue

            if raw.empty:
                continue

            # 単一銘柄の場合はカラムが 1 レベル
            if len(batch) == 1:
                ticker = batch[0]
                for dt, row in raw.iterrows():
                    adj = row.get("Adj Close") or row.get("Close")
                    if adj is None or pd.isna(adj):
                        continue
                    records.append(
                        {
                            "ticker": ticker,
                            "date": dt.date(),
                            "open": _safe_float(row.get("Open")),
                            "high": _safe_float(row.get("High")),
                            "low": _safe_float(row.get("Low")),
                            "close": _safe_float(row.get("Close")),
                            "adj_close": float(adj),
                            "volume": _safe_int(row.get("Volume")),
                        }
                    )
            else:
                adj_close = raw.get("Adj Close", raw.get("Close"))
                close = raw.get("Close", adj_close)
                open_ = raw.get("Open")
                high = raw.get("High")
                low = raw.get("Low")
                volume = raw.get("Volume")

                for dt, row in adj_close.iterrows():
                    for ticker in batch:
                        if ticker not in row.index:
                            continue
                        adj = row[ticker]
                        if pd.isna(adj):
                            continue
                        records.append(
                            {
                                "ticker": ticker,
                                "date": dt.date(),
                                "open": _safe_float(open_[ticker][dt] if open_ is not None else None),
                                "high": _safe_float(high[ticker][dt] if high is not None else None),
                                "low": _safe_float(low[ticker][dt] if low is not None else None),
                                "close": _safe_float(close[ticker][dt] if close is not None else None),
                                "adj_close": float(adj),
                                "volume": _safe_int(volume[ticker][dt] if volume is not None else None),
                            }
                        )

        logger.info("yfinance: %d レコード取得完了", len(records))
        return records

    def fetch_stock_list(self) -> list[dict[str, Any]]:
        """デフォルトリストを返す（yfinance では銘柄一覧 API なし）"""
        return [
            {
                "ticker": t,
                "name": t,
                "sector_33": None,
                "sector_17": None,
                "market": "Prime",
                "lot_size": 100,
            }
            for t in _DEFAULT_JP_TICKERS
        ]

    @staticmethod
    def default_tickers() -> list[str]:
        return list(_DEFAULT_JP_TICKERS)


def _safe_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None and not pd.isna(v) else None
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None and not pd.isna(v) else None
    except (TypeError, ValueError):
        return None
