from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PriceProcessor:
    """株価データの前処理・クレンジング"""

    def __init__(self, max_missing_rate: float = 0.10) -> None:
        self._max_missing_rate = max_missing_rate

    def clean(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        調整済終値の DataFrame をクレンジングして返す。
        - 欠損率が max_missing_rate を超える銘柄を除外
        - 残りの欠損を前日値で補完
        - 0 以下の値を NaN に置換
        """
        if price_df.empty:
            return price_df

        # 0 以下を NaN 化
        df = price_df.copy()
        df[df <= 0] = np.nan

        # 欠損率チェック
        missing_rate = df.isna().mean()
        valid_tickers = missing_rate[missing_rate <= self._max_missing_rate].index.tolist()
        removed = set(df.columns) - set(valid_tickers)
        if removed:
            logger.info("欠損率超過のため除外: %s", removed)
        df = df[valid_tickers]

        # 前日値で補完 (ffill) → 残り先頭欠損は後続値で補完 (bfill)
        df = df.ffill().bfill()

        return df

    def validate(self, records: list[dict]) -> list[dict]:
        """DB 保存前の個別レコードバリデーション"""
        valid = []
        for r in records:
            if r.get("adj_close") is None or float(r["adj_close"]) <= 0:
                continue
            valid.append(r)
        return valid
