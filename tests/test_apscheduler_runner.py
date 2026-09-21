import pytest

import scripts.apscheduler_runner as runner


def test_load_schedule_config_reads_environment(monkeypatch):
	monkeypatch.setenv("CRON_HOUR", "11")
	monkeypatch.setenv("CRON_MINUTE", "10")

	assert runner.load_schedule_config() == runner.ScheduleConfig(hour=11, minute=10)


def test_load_schedule_config_requires_environment_values(monkeypatch):
	monkeypatch.setattr(runner, "load_dotenv", lambda: None)
	monkeypatch.delenv("CRON_HOUR", raising=False)
	monkeypatch.delenv("CRON_MINUTE", raising=False)

	with pytest.raises(ValueError, match="required"):
		runner.load_schedule_config()


def test_create_scheduler_uses_ist_and_daily_time(monkeypatch):
	created = {}

	class FakeScheduler:
		def __init__(self, **kwargs):
			created["timezone"] = kwargs["timezone"]

		def add_job(self, func, **kwargs):
			created["func"] = func
			created["job"] = kwargs

	monkeypatch.setattr(runner, "BackgroundScheduler", FakeScheduler)

	runner.create_scheduler(runner.ScheduleConfig(hour=11, minute=10))

	assert created["timezone"] == runner.SCHEDULER_TIMEZONE
	assert created["timezone"].key == "Asia/Kolkata"
	assert created["func"] is runner.run_daily
	assert created["job"]["max_instances"] == 1
	assert created["job"]["coalesce"] is True
	trigger = created["job"]["trigger"]
	assert trigger.timezone.key == "Asia/Kolkata"
	assert str(trigger.fields[5]) == "11"
	assert str(trigger.fields[6]) == "10"


def test_start_scheduler_is_idempotent(monkeypatch):
	created = []

	class FakeScheduler:
		running = False

		def start(self):
			created.append(self)
			self.running = True

	monkeypatch.setattr(runner, "create_scheduler", lambda: FakeScheduler())
	runner._scheduler = None

	try:
		first = runner.start_scheduler()
		second = runner.start_scheduler()
		assert first is second
		assert len(created) == 1
	finally:
		runner._scheduler = None


def test_shutdown_scheduler_stops_and_clears_scheduler():
	class FakeScheduler:
		running = True

		def shutdown(self, wait=True):
			self.running = False

	runner._scheduler = FakeScheduler()
	try:
		runner.shutdown_scheduler()
		assert runner._scheduler is None
	finally:
		runner._scheduler = None


@pytest.mark.parametrize(
	"name,value",
	[
		("CRON_HOUR", "24"),
		("CRON_MINUTE", "60"),
	],
)
def test_load_schedule_config_rejects_invalid_ranges(monkeypatch, name, value):
	monkeypatch.setenv(name, value)

	with pytest.raises(ValueError):
		runner.load_schedule_config()