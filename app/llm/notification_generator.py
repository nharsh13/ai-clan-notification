import asyncio
import json
import logging
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.llm.generate_engagement_sentiment_notification import (
    build_engagement_sentiment_notification_prompt,
    validate_engagement_sentiment_notification,
)
from app.llm.generate_performance_notification import (
    build_performance_notification_prompt,
    validate_performance_notification,
)
from app.llm.generate_qa_sentiment_notification import (
    build_qa_sentiment_notification_prompt,
    validate_qa_sentiment_notification,
)

MODEL = OPENAI_MODEL
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (1, 2, 4)
logger = logging.getLogger(__name__)


def _is_temporary_openai_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True
    return isinstance(error, APIStatusError) and error.status_code >= 500


def _generate_with_retry(client, model: str, prompt: str) -> Any:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.responses.create(model=model, input=prompt)
        except Exception as error:
            if not _is_temporary_openai_error(error) or attempt == MAX_RETRIES:
                raise
            delay = RETRY_BACKOFF_SECONDS[attempt]
            logger.warning(
                "Temporary OpenAI error; retrying in %ss (attempt %s/%s) error_type=%s",
                delay,
                attempt + 1,
                MAX_RETRIES,
                type(error).__name__,
            )
            time.sleep(delay)


async def _generate_with_retry_async(client, model: str, prompt: str) -> Any:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await client.responses.create(model=model, input=prompt)
        except Exception as error:
            if not _is_temporary_openai_error(error) or attempt == MAX_RETRIES:
                raise
            delay = RETRY_BACKOFF_SECONDS[attempt]
            logger.warning(
                "Temporary OpenAI error; retrying in %ss (attempt %s/%s) error_type=%s",
                delay,
                attempt + 1,
                MAX_RETRIES,
                type(error).__name__,
            )
            await asyncio.sleep(delay)


def _get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError(
            "Missing required environment variable: OPENAI_API_KEY. Add it to the .env file."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def _get_async_openai_client() -> AsyncOpenAI:
    if not OPENAI_API_KEY:
        raise ValueError(
            "Missing required environment variable: OPENAI_API_KEY. Add it to the .env file."
        )
    return AsyncOpenAI(api_key=OPENAI_API_KEY)


def _parse_json_response(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM returned invalid JSON") from exc


def generate_engagement_sentiment_notification(
    user_name: str,
    language: str,
    response_data: dict,
) -> dict | None:
    if not response_data:
        return None

    client = _get_openai_client()
    prompt = build_engagement_sentiment_notification_prompt(
        user_name=user_name,
        language=language,
        response_data=response_data,
    )
    response = _generate_with_retry(client, MODEL, prompt)
    content = response.output_text.strip()
    notification = _parse_json_response(content)
    validate_engagement_sentiment_notification(notification)
    return notification


def generate_qa_sentiment_notification(
    user_name: str,
    language: str,
    prepared_qa: dict,
) -> dict:
    client = _get_openai_client()
    prompt = build_qa_sentiment_notification_prompt(
        user_name=user_name,
        language=language,
        prepared_qa=prepared_qa,
    )
    response = _generate_with_retry(client, MODEL, prompt)
    content = response.output_text.strip()
    notification = _parse_json_response(content)
    validate_qa_sentiment_notification(notification)
    return notification


def generate_performance_notification(
    user_name: str,
    language: str,
    weakest_kii: dict,
    video: dict,
) -> dict:
    client = _get_openai_client()
    prompt = build_performance_notification_prompt(
        user_name=user_name,
        language=language,
        weakest_kii=weakest_kii,
        video=video,
    )
    response = _generate_with_retry(client, MODEL, prompt)
    notification = _parse_json_response(response.output_text.strip())
    validate_performance_notification(notification)
    return notification


class NotificationGenerator:
    def __init__(self, client=None, model: str = MODEL):
        self.client = client
        self.model = model

    @staticmethod
    def validate(payload) -> dict[str, str]:
        if isinstance(payload, str):
            payload = _parse_json_response(payload)
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be an object")
        title = payload.get("title")
        description = payload.get("description")
        if not isinstance(title, str) or not title.strip() or not isinstance(description, str) or not description.strip():
            raise ValueError("LLM response requires non-empty title and description")
        action = payload.get("action", "Watch now")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("LLM response action must be a non-empty string")
        return {
            "title": title.strip(),
            "description": description.strip(),
            "action": action.strip(),
        }

    def generate(self, prompt: str) -> dict[str, str]:
        client = self.client or _get_openai_client()
        response = _generate_with_retry(client, self.model, prompt)
        return self.validate(response.output_text)

    async def generate_async(self, prompt: str) -> dict[str, str]:
        client = self.client or _get_async_openai_client()
        response = await _generate_with_retry_async(client, self.model, prompt)
        return self.validate(response.output_text)


__all__ = [
    "MAX_RETRIES",
    "MODEL",
    "RETRY_BACKOFF_SECONDS",
    "NotificationGenerator",
    "_generate_with_retry",
    "_get_openai_client",
    "_is_temporary_openai_error",
    "_parse_json_response",
    "build_engagement_sentiment_notification_prompt",
    "build_qa_sentiment_notification_prompt",
    "build_performance_notification_prompt",
    "generate_engagement_sentiment_notification",
    "generate_qa_sentiment_notification",
    "generate_performance_notification",
    "validate_engagement_sentiment_notification",
    "validate_qa_sentiment_notification",
    "validate_performance_notification",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OpenAI",
]
