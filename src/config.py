from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # J-Quants API
    jquants_mail: str = ""
    jquants_password: str = ""
    jquants_refresh_token: str = ""

    # データベース
    database_url: str = "postgresql://mvopt:mvopt@localhost:5432/mvopt"

    # キャッシュディレクトリ
    cache_dir: str = ".cache"

    # 最適化デフォルトパラメータ
    default_lookback_days: int = 252
    default_covariance_method: str = "ledoit_wolf"
    default_risk_free_rate: float = 0.001
    default_max_stocks: int = 100
    default_min_market_cap: int = 10_000_000_000      # 100億円
    default_min_avg_volume: int = 100_000             # 10万株

    # スケジューラ
    data_update_cron: str = "30 17 * * 1-5"

    # ログ
    log_level: str = "INFO"


settings = Settings()
