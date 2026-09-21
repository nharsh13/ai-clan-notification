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

	monkeypatch.setattr(runner, "BlockingScheduler", FakeScheduler)

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