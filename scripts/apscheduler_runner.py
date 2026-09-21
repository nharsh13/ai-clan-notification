import logging
import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from scripts.run_daily import main as run_daily

logger = logging.getLogger(__name__)
SCHEDULER_TIMEZONE = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class ScheduleConfig:
	hour: int
	minute: int


def load_schedule_config() -> ScheduleConfig:
	load_dotenv()
	hour_value = os.getenv("CRON_HOUR")
	minute_value = os.getenv("CRON_MINUTE")
	if hour_value is None or minute_value is None:
		raise ValueError("CRON_HOUR and CRON_MINUTE are required")
	try:
		hour = int(hour_value)
		minute = int(minute_value)
	except ValueError as exc:
		raise ValueError("CRON_HOUR and CRON_MINUTE must be integers") from exc

	if not 0 <= hour <= 23:
		raise ValueError("CRON_HOUR must be between 0 and 23")
	if not 0 <= minute <= 59:
		raise ValueError("CRON_MINUTE must be between 0 and 59")

	return ScheduleConfig(hour=hour, minute=minute)


def create_scheduler(config: ScheduleConfig | None = None) -> BlockingScheduler:
	schedule = config or load_schedule_config()
	scheduler = BlockingScheduler(timezone=SCHEDULER_TIMEZONE)
	scheduler.add_job(
		run_daily,
		trigger=CronTrigger(
			hour=schedule.hour,
			minute=schedule.minute,
			timezone=SCHEDULER_TIMEZONE,
		),
		id="daily_notification_scheduler",
		max_instances=1,
		coalesce=True,
		misfire_grace_time=3600,
	)
	return scheduler


def main() -> None:
	config = load_schedule_config()
	logger.info(
		"Starting APScheduler timezone=%s hour=%s minute=%s",
		SCHEDULER_TIMEZONE,
		config.hour,
		config.minute,
	)
	create_scheduler(config).start()


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	main()