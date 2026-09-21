import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")


class ConfigurationError(ValueError):
	"""Raised when the application cannot run with the current environment."""


@dataclass(frozen=True)
class Settings:
	database_url: str
	openai_api_key: str
	openai_model: str
	embedding_provider: str
	embedding_api_key: str
	embedding_model: str
	notification_send_url: str
	notification_timeout_seconds: float
	video_deep_link_template: str
	openai_timeout_seconds: float = 60


def load_settings() -> Settings:
	"""Load configuration from environment variables and the local .env file."""

	openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
	embedding_provider = os.getenv(
		"EMBED_PROVIDER",
		os.getenv(
			"EMBEDDING_PROVIDER",
		"openai" if openai_api_key else "sentence_transformers",
		),
	).strip().lower()
	timeout = os.getenv("REMOTE_NOTIFICATION_TIMEOUT_SECONDS", "30").strip()
	openai_timeout = os.getenv("OPENAI_TIMEOUT_SECONDS", "60").strip()

	try:
		timeout_seconds = float(timeout)
		openai_timeout_seconds = float(openai_timeout)
	except ValueError as exc:
		raise ConfigurationError(
			"REMOTE_NOTIFICATION_TIMEOUT_SECONDS and OPENAI_TIMEOUT_SECONDS must be numbers"
		) from exc

	default_embedding_model = (
		"all-MiniLM-L6-v2"
		if embedding_provider == "sentence_transformers"
		else "text-embedding-3-small"
	)

	return Settings(
		database_url=os.getenv("DATABASE_URL", "").strip(),
		openai_api_key=openai_api_key,
		openai_model=os.getenv("OPENAI_MODEL", "gpt-5-nano").strip(),
		embedding_provider=embedding_provider,
		embedding_api_key=(
			os.getenv("EMBEDDING_API_KEY", "").strip() or openai_api_key
		),
		embedding_model=os.getenv(
			"EMBED_MODEL",
			os.getenv("EMBEDDING_MODEL", default_embedding_model),
		).strip(),
		notification_send_url=os.getenv("REMOTE_NOTIFICATION_SEND_URL", "").strip(),
		notification_timeout_seconds=timeout_seconds,
		video_deep_link_template=os.getenv(
			"VIDEO_DEEP_LINK_TEMPLATE", "/videos/{video_id}"
		).strip(),
		openai_timeout_seconds=openai_timeout_seconds,
	)


def validate_configuration(
	settings: Settings | None = None,
	*,
	require_notification_sender: bool = False,
) -> Settings:
	"""Validate runtime requirements without exposing secret values."""

	current = settings or load_settings()
	missing = []
	if not current.database_url:
		missing.append("DATABASE_URL")
	if not current.openai_api_key:
		missing.append("OPENAI_API_KEY")
	if require_notification_sender and not current.notification_send_url:
		missing.append("REMOTE_NOTIFICATION_SEND_URL")
	if current.embedding_provider not in {"openai", "sentence_transformers"}:
		raise ConfigurationError(
			"EMBED_PROVIDER must be 'openai' or 'sentence_transformers'"
		)
	if (
		current.embedding_provider == "openai"
		and current.embedding_model.lower() == "all-minilm-l6-v2"
	):
		raise ConfigurationError(
			"all-MiniLM-L6-v2 is a Sentence Transformers model and cannot be used "
			"with EMBED_PROVIDER=openai; use EMBED_PROVIDER=sentence_transformers"
		)
	if current.embedding_provider == "openai" and not current.embedding_api_key:
		missing.append("EMBEDDING_API_KEY or OPENAI_API_KEY")
	if "{video_id}" not in current.video_deep_link_template:
		raise ConfigurationError(
			"VIDEO_DEEP_LINK_TEMPLATE must contain {video_id}"
		)
	if missing:
		raise ConfigurationError(
			"Missing required environment variable(s): " + ", ".join(missing)
		)
	return current


_settings = load_settings()

# Backward-compatible names used by the existing modules.
DATABASE_URL = _settings.database_url
OPENAI_API_KEY = _settings.openai_api_key
OPENAI_MODEL = _settings.openai_model
EMBEDDING_PROVIDER = _settings.embedding_provider
EMBEDDING_API_KEY = _settings.embedding_api_key
EMBEDDING_MODEL = _settings.embedding_model
REMOTE_NOTIFICATION_SEND_URL = _settings.notification_send_url
REMOTE_NOTIFICATION_TIMEOUT_SECONDS = _settings.notification_timeout_seconds
VIDEO_DEEP_LINK_TEMPLATE = _settings.video_deep_link_template
OPENAI_TIMEOUT_SECONDS = _settings.openai_timeout_seconds
