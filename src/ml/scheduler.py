from apscheduler.schedulers.blocking import BlockingScheduler
from src.ml.train import train
import logging

logging.basicConfig(level=logging.INFO)
scheduler = BlockingScheduler()

@scheduler.scheduled_job("cron", day_of_week="sat", hour=12, minute=0)
def scheduled_training():
    logging.info("Saturday noon — starting model retraining...")
    train()
    logging.info("Retraining complete.")

if __name__ == "__main__":
    logging.info("Scheduler started. Waiting for Saturday 12:00...")
    scheduler.start()