from __future__ import annotations

import logging
from datetime import date, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import settings

logger = logging.getLogger(__name__)


def update_prices_job() -> None:
    """株価データの定期更新ジョブ"""
    logger.info("株価データ更新ジョブ開始")
    try:
        from src.data.database import SessionLocal
        from src.data.fetchers.yfinance import YFinanceFetcher
        from src.data.processors.price import PriceProcessor
        from src.data.repository import PriceRepository, StockRepository

        db = SessionLocal()
        try:
            stock_repo = StockRepository(db)
            price_repo = PriceRepository(db)

            stocks = stock_repo.get_all_active()
            tickers = [s.ticker for s in stocks]

            if not tickers:
                from src.data.fetchers.yfinance import YFinanceFetcher
                tickers = YFinanceFetcher.default_tickers()

            fetcher = YFinanceFetcher()
            processor = PriceProcessor()

            end = date.today()
            start = end - timedelta(days=5)

            records = fetcher.fetch_prices(tickers, start, end)
            valid_records = processor.validate(records)

            if valid_records:
                price_repo.upsert_prices(valid_records)
                logger.info("株価データ更新完了: %d レコード", len(valid_records))
            else:
                logger.warning("有効なレコードが 0 件")
        finally:
            db.close()
    except Exception as e:
        logger.error("株価データ更新ジョブエラー: %s", e, exc_info=True)


def update_stock_list_job() -> None:
    """銘柄マスタの定期更新ジョブ（週次）"""
    logger.info("銘柄マスタ更新ジョブ開始")
    try:
        from src.data.database import SessionLocal
        from src.data.repository import StockRepository

        db = SessionLocal()
        try:
            # J-Quants が利用可能であれば使用、そうでなければスキップ
            if settings.jquants_mail:
                from src.data.fetchers.jquants import JQuantsFetcher
                fetcher = JQuantsFetcher()
            else:
                from src.data.fetchers.yfinance import YFinanceFetcher
                fetcher = YFinanceFetcher()

            stock_list = fetcher.fetch_stock_list()
            if stock_list:
                repo = StockRepository(db)
                repo.upsert_stocks(stock_list)
                logger.info("銘柄マスタ更新完了: %d 銘柄", len(stock_list))
        finally:
            db.close()
    except Exception as e:
        logger.error("銘柄マスタ更新ジョブエラー: %s", e, exc_info=True)


def create_scheduler() -> BackgroundScheduler:
    """スケジューラを生成・設定して返す"""
    scheduler = BackgroundScheduler()

    # 株価データ更新（平日 17:30）
    cron_parts = settings.data_update_cron.split()
    if len(cron_parts) == 5:
        minute, hour, dom, month, dow = cron_parts
        scheduler.add_job(
            update_prices_job,
            CronTrigger(minute=minute, hour=hour, day=dom, month=month, day_of_week=dow),
            id="update_prices",
            name="株価データ更新",
            replace_existing=True,
        )

    # 銘柄マスタ更新（毎週月曜 8:00）
    scheduler.add_job(
        update_stock_list_job,
        CronTrigger(day_of_week="mon", hour=8, minute=0),
        id="update_stock_list",
        name="銘柄マスタ更新",
        replace_existing=True,
    )

    return scheduler


if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    scheduler = create_scheduler()
    scheduler.start()
    logger.info("スケジューラ起動完了")
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
