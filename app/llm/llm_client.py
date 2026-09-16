import json
import logging
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from app.config import OPENAI_API_KEY, OPENAI_MODEL

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
                "Temporary OpenAI error; retrying in %ss (attempt %s/%s): %s",
                delay,
                attempt + 1,
                MAX_RETRIES,
                error,
            )
            time.sleep(delay)


def _get_openai_client() -> OpenAI:
    """
    Create OpenAI client.
    """

    if not OPENAI_API_KEY:
        raise ValueError(
            "Missing required environment variable: OPENAI_API_KEY. "
            "Add it to the .env file."
        )

    return OpenAI(api_key=OPENAI_API_KEY)


def _parse_json_response(content: str) -> dict:
    """
    Parse JSON returned by the LLM.
    """

    try:
        return json.loads(content)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {content}"
        ) from exc


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
