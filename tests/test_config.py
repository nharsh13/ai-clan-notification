import pytest

import app.config as config
import app.llm.notification_generator as notification_generator


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

def test_load_settings_reads_environment(monkeypatch):
    """
    Verify that load_settings() correctly reads environment
    variables.
    """

    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://db.example/clan",
    )
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "sk-test",
    )
    monkeypatch.setenv(
        "EMBED_PROVIDER",
        "openai",
    )
    monkeypatch.setenv(
        "EMBEDDING_API_KEY",
        "sk-embedding-test",
    )
    monkeypatch.setenv(
        "REMOTE_NOTIFICATION_SEND_URL",
        "https://notify.example/send",
    )

    settings = config.load_settings()

    assert settings.database_url == (
        "postgresql://db.example/clan"
    )

    assert settings.openai_api_key == "sk-test"

    assert settings.embedding_api_key == (
        "sk-embedding-test"
    )

    assert settings.notification_send_url == (
        "https://notify.example/send"
    )


# ============================================================
# SENTENCE TRANSFORMERS CONFIGURATION
# ============================================================

def test_sentence_transformers_configuration_is_valid(
    monkeypatch,
):
    """
    Verify Sentence Transformers configuration is accepted.
    """

    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://db.example/clan",
    )
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "sk-test",
    )
    monkeypatch.setenv(
        "EMBED_PROVIDER",
        "sentence_transformers",
    )
    monkeypatch.setenv(
        "EMBED_MODEL",
        "all-MiniLM-L6-v2",
    )

    settings = config.load_settings()

    validated = config.validate_configuration(settings)

    assert validated.embedding_model == (
        "all-MiniLM-L6-v2"
    )


# ============================================================
# INVALID OPENAI + SENTENCE TRANSFORMER MODEL
# ============================================================

def test_openai_rejects_sentence_transformers_model():
    """
    OpenAI provider must not use a Sentence Transformer model.
    """

    settings = config.Settings(
        database_url="postgresql://db.example/clan",
        openai_api_key="sk-test",
        openai_model="gpt-5-nano",
        embedding_provider="openai",
        embedding_api_key="sk-test",
        embedding_model="all-MiniLM-L6-v2",
        notification_send_url="",
        notification_timeout_seconds=30,
        video_deep_link_template="/videos/{video_id}",
    )

    with pytest.raises(
        config.ConfigurationError,
        match="Sentence Transformers",
    ):
        config.validate_configuration(settings)


# ============================================================
# UNSUPPORTED EMBEDDING PROVIDER
# ============================================================

def test_unsupported_embedding_provider_has_clear_error():
    """
    Unsupported embedding providers must raise a clear error.
    """

    settings = config.Settings(
        database_url="postgresql://db.example/clan",
        openai_api_key="sk-test",
        openai_model="gpt-5-nano",
        embedding_provider="unknown",
        embedding_api_key="sk-test",
        embedding_model="text-embedding-3-small",
        notification_send_url="",
        notification_timeout_seconds=30,
        video_deep_link_template="/videos/{video_id}",
    )

    with pytest.raises(
        config.ConfigurationError,
        match="EMBED_PROVIDER",
    ):
        config.validate_configuration(settings)


# ============================================================
# MISSING OPENAI API KEY
# ============================================================

def test_missing_api_key_has_clear_error():
    """
    Verify that missing OPENAI_API_KEY is rejected.
    """

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

    with pytest.raises(
        config.ConfigurationError,
        match="OPENAI_API_KEY",
    ):
        config.validate_configuration(settings)


# ============================================================
# OPENAI CLIENT API KEY
# ============================================================

def test_llm_client_receives_configured_api_key(
    monkeypatch,
):
    """
    Verify that the configured OPENAI_API_KEY is passed
    to the OpenAI client.
    """

    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key):
            captured["api_key"] = api_key

    monkeypatch.setattr(
        notification_generator,
        "OPENAI_API_KEY",
        "sk-configured",
    )

    monkeypatch.setattr(
        notification_generator,
        "OpenAI",
        FakeOpenAI,
    )

    notification_generator._get_openai_client()

    assert captured["api_key"] == "sk-configured"