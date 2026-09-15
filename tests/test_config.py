import importlib

import pytest

import app.config as config
import app.llm.notification_generator as notification_generator


def test_load_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://db.example/clan")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-embedding-test")
    monkeypatch.setenv("REMOTE_NOTIFICATION_SEND_URL", "https://notify.example/send")

    settings = config.load_settings()

    assert settings.database_url == "postgresql://db.example/clan"
    assert settings.openai_api_key == "sk-test"
    assert settings.embedding_api_key == "sk-embedding-test"
    assert settings.notification_send_url == "https://notify.example/send"


def test_missing_api_key_has_clear_error(monkeypatch):
    settings = config.Settings(
        database_url="postgresql://db.example/clan",
        openai_api_key="",
        openai_model="gpt-test",
        embedding_provider="sentence_transformers",
        embedding_api_key="",
        embedding_model="all-MiniLM-L6-v2",
        notification_send_url="",
        notification_timeout_seconds=30,
        video_deep_link_template="/videos/{video_id}",
    )

    with pytest.raises(config.ConfigurationError, match="OPENAI_API_KEY"):
        config.validate_configuration(settings)


def test_llm_client_receives_configured_api_key(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key):
            captured["api_key"] = api_key

    monkeypatch.setattr(notification_generator, "OPENAI_API_KEY", "sk-configured")
    monkeypatch.setattr(notification_generator, "OpenAI", FakeOpenAI)

    notification_generator._get_openai_client()

    assert captured["api_key"] == "sk-configured"
