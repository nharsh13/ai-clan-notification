import asyncio

from app import main as app_module


def test_fastapi_lifespan_starts_and_stops_scheduler(monkeypatch):
	calls = []
	monkeypatch.setattr(app_module, "validate_configuration", lambda: None)
	monkeypatch.setattr(app_module, "start_scheduler", lambda: calls.append("start"))
	monkeypatch.setattr(app_module, "shutdown_scheduler", lambda: calls.append("shutdown"))

	async def exercise():
		async with app_module.lifespan(app_module.app):
			assert calls == ["start"]

	asyncio.run(exercise())

	assert calls == ["start", "shutdown"]