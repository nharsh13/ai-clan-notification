import logging
import os
from datetime import date, datetime, time
from threading import Event, Lock
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from scripts.run_daily import main as run_daily

logger = logging.getLogger(__name__)
SCHEDULER_TIMEZONE = ZoneInfo("Asia/Kolkata")
_scheduler_lock = Lock()
_scheduler: BackgroundScheduler | None = None


@dataclass(frozen=True)
class ScheduleConfig:
	start_date: datetime
	hour: int
	minute: int


def load_schedule_config() -> ScheduleConfig:
	load_dotenv()
	start_date_value = os.getenv("CRON_START_DATE")
	hour_value = os.getenv("CRON_HOUR")
	minute_value = os.getenv("CRON_MINUTE")
	if start_date_value is None:
		raise ValueError("CRON_START_DATE is required (format: YYYY-MM-DD)")
	if hour_value is None or minute_value is None:
		raise ValueError("CRON_HOUR and CRON_MINUTE are required")
	try:
		if len(start_date_value) != 10 or start_date_value[4] != "-" or start_date_value[7] != "-":
			raise ValueError
		start_date = date.fromisoformat(start_date_value)
	except ValueError as exc:
		raise ValueError("CRON_START_DATE must be a valid date in YYYY-MM-DD format") from exc
	try:
		hour = int(hour_value)
		minute = int(minute_value)
	except ValueError as exc:
		raise ValueError("CRON_HOUR and CRON_MINUTE must be integers") from exc

	if not 0 <= hour <= 23:
		raise ValueError("CRON_HOUR must be between 0 and 23")
	if not 0 <= minute <= 59:
		raise ValueError("CRON_MINUTE must be between 0 and 59")

	return ScheduleConfig(
		start_date=datetime.combine(start_date, time.min, tzinfo=SCHEDULER_TIMEZONE),
		hour=hour,
		minute=minute,
	)


def create_scheduler(config: ScheduleConfig | None = None) -> BackgroundScheduler:
	schedule = config or load_schedule_config()
	scheduler = BackgroundScheduler(timezone=SCHEDULER_TIMEZONE)
	scheduler.add_job(
		run_daily,
		trigger=CronTrigger(
			hour=schedule.hour,
			minute=schedule.minute,
			start_date=schedule.start_date,
			timezone=SCHEDULER_TIMEZONE,
		),
		id="daily_notification_scheduler",
		max_instances=1,
		coalesce=True,
		misfire_grace_time=3600,
	)
	return scheduler


def start_scheduler() -> BackgroundScheduler:
	global _scheduler
	with _scheduler_lock:
		if _scheduler is None:
			_scheduler = create_scheduler()
		if not _scheduler.running:
			_scheduler.start()
			job = (
				_scheduler.get_job("daily_notification_scheduler")
				if hasattr(_scheduler, "get_job")
				else None
			)
			logger.info("[SCHEDULER] Scheduler status: STARTED")
			logger.info("[SCHEDULER] Current date/time in IST: %s", datetime.now(SCHEDULER_TIMEZONE).isoformat())
			logger.info("[SCHEDULER] Configured schedule: %s", job.trigger if job else "unavailable")
			logger.info("[SCHEDULER] Job ID: daily_notification_scheduler")
			logger.info("[SCHEDULER] Exact next scheduled run time: %s", job.next_run_time if job else "unavailable")
			logger.info("[SCHEDULER] Status: RUNNING")
		return _scheduler


def shutdown_scheduler() -> None:
	global _scheduler
	with _scheduler_lock:
		if _scheduler is not None and _scheduler.running:
			_scheduler.shutdown(wait=True)
			logger.info("[SCHEDULER] Scheduler STOPPED")
			logger.info("[SCHEDULER] Shutdown completed")
		_scheduler = None


def main() -> None:
	config = load_schedule_config()
	logger.info(
		"Starting APScheduler timezone=%s hour=%s minute=%s",
		SCHEDULER_TIMEZONE,
		config.hour,
		config.minute,
	)
	scheduler = create_scheduler(config)
	try:
		scheduler.start()
		job = scheduler.get_job("daily_notification_scheduler")
		logger.info("[SCHEDULER] Scheduler status: STARTED")
		logger.info("[SCHEDULER] Current date/time in IST: %s", datetime.now(SCHEDULER_TIMEZONE).isoformat())
		logger.info("[SCHEDULER] Configured schedule: daily at %02d:%02d IST", config.hour, config.minute)
		logger.info("[SCHEDULER] Job ID: daily_notification_scheduler")
		logger.info("[SCHEDULER] Exact next scheduled run time: %s", job.next_run_time if job else "unavailable")
		logger.info("[SCHEDULER] Status: RUNNING")
		Event().wait()
	except KeyboardInterrupt:
		logger.info("[SCHEDULER] Ctrl+C received; stopping scheduler")
	except Exception:
		logger.exception("[ERROR] Unexpected scheduler exception")
		raise
	finally:
		scheduler.shutdown(wait=True)
		logger.info("[SCHEDULER] Scheduler STOPPED")
		logger.info("[SCHEDULER] Shutdown completed")


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	main()
