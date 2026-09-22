import asyncio
import logging
import time
from typing import Any, Dict

import httpx
import requests

from app.config import REMOTE_NOTIFICATION_SEND_URL, REMOTE_NOTIFICATION_TIMEOUT_SECONDS

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (1, 2, 4)
logger = logging.getLogger(__name__)


class TemporaryNotificationSenderError(Exception):
    pass

class NotificationSender:
    def __init__(self, remote_url: str | None = None):
        self.remote_url: str = (
            remote_url
            or REMOTE_NOTIFICATION_SEND_URL
            or ""
        )

    def _prepare_payload(
        self,
        user_id: int,
        notification_type: str,
        title: str,
        description: str | None,
        reference_id: int,
        video_popup: bool | str | None = False,
        image: str | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(video_popup, str):
            video_popup = video_popup.strip().upper() in {"Y", "YES", "TRUE", "1"}
        else:
            video_popup = bool(video_popup)

        return [{
            "user_id": user_id,
            "title": title,
            "description": description,
            "notification_type": notification_type,
            "reference_id": reference_id,
            "video_popup": video_popup,
            "image": image,
        }]

    def send(
        self,
        user_id: int,
        notification_type: str,
        title: str,
        description: str | None,
        reference_id: int,
        video_popup: bool | str | None = False,
        image: str | None = None,
    ) -> Dict[str, Any]:
        if not self.remote_url:
            raise ValueError(
                "REMOTE_NOTIFICATION_SEND_URL is not configured"
            )

        payload = self._prepare_payload(
            user_id,
            notification_type,
            title,
            description,
            reference_id,
            video_popup,
            image,
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(
                    self.remote_url,
                    json=payload,
                    timeout=REMOTE_NOTIFICATION_TIMEOUT_SECONDS,
                )
                if not response.ok:
                    if response.status_code == 429 or response.status_code >= 500:
                        raise TemporaryNotificationSenderError(
                            f"Remote API error {response.status_code}: {response.text}"
                        )
                    raise ValueError(
                        f"Remote API error {response.status_code}: {response.text}"
                    )
                break
            except (requests.Timeout, requests.ConnectionError, TemporaryNotificationSenderError) as error:
                if attempt == MAX_RETRIES:
                    raise
                delay = RETRY_BACKOFF_SECONDS[attempt]
                logger.warning(
                    "Temporary notification sender error; retrying in %ss (attempt %s/%s): %s",
                    delay,
                    attempt + 1,
                    MAX_RETRIES,
                    error,
                )
                time.sleep(delay)

        try:
            response_body = response.json()
        except ValueError:
            response_body = {
                "message": response.text
            }

        return {
            "remote_url": self.remote_url,
            "request_payload": payload,
            "response": response_body,
            "status_code": response.status_code,
        }

    async def send_async(
        self,
        user_id: int,
        notification_type: str,
        title: str,
        description: str | None,
        reference_id: int,
        video_popup: bool | str | None = False,
        image: str | None = None,
    ) -> Dict[str, Any]:
        if not self.remote_url:
            raise ValueError(
                "REMOTE_NOTIFICATION_SEND_URL is not configured"
            )

        payload = self._prepare_payload(
            user_id,
            notification_type,
            title,
            description,
            reference_id,
            video_popup,
            image,
        )

        async with httpx.AsyncClient(timeout=REMOTE_NOTIFICATION_TIMEOUT_SECONDS) as client:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = await client.post(
                        self.remote_url,
                        json=payload,
                    )
                    is_error = getattr(response, "is_error", not getattr(response, "ok", False))
                    if not is_error:
                        break
                    if response.status_code == 429 or response.status_code >= 500:
                        raise TemporaryNotificationSenderError(
                            f"Remote API error {response.status_code}: {response.text}"
                        )
                    raise ValueError(
                        f"Remote API error {response.status_code}: {response.text}"
                    )
                except (httpx.TimeoutException, httpx.RequestError, TemporaryNotificationSenderError) as error:
                    if attempt == MAX_RETRIES:
                        raise
                    delay = RETRY_BACKOFF_SECONDS[attempt]
                    logger.warning(
                        "Temporary notification sender error; retrying in %ss (attempt %s/%s): %s",
                        delay,
                        attempt + 1,
                        MAX_RETRIES,
                        error,
                    )
                    await asyncio.sleep(delay)

        try:
            response_json = response.json()
            if asyncio.iscoroutine(response_json):
                response_body = await response_json
            else:
                response_body = response_json
        except ValueError:
            response_body = {"message": response.text}

        return {
            "remote_url": self.remote_url,
            "request_payload": payload,
            "response": response_body,
            "status_code": response.status_code,
        }