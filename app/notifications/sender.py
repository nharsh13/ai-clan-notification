from typing import Any, Dict
import requests

from app.config import REMOTE_NOTIFICATION_SEND_URL, REMOTE_NOTIFICATION_TIMEOUT_SECONDS

class NotificationSender:
    def __init__(self, remote_url: str | None = None):
        self.remote_url: str = (
            remote_url
            or REMOTE_NOTIFICATION_SEND_URL
            or ""
        )

    def send(
        self,
        user_id: int,
        notification_type: str,
        title: str,
        description: str,
        reference_id: int,
        video_popup: str = "N",
    ) -> Dict[str, Any]:
        if not self.remote_url:
            raise ValueError(
                "REMOTE_NOTIFICATION_SEND_URL is not configured"
            )

        payload = [
            {
                "description": description,
                "notification_type": notification_type,
                "reference_id": reference_id,
                "title": title,
                "user_id": user_id,
                "video_popup": video_popup,
            }
        ]

        response = requests.post(
            self.remote_url,
            json=payload,
            timeout=REMOTE_NOTIFICATION_TIMEOUT_SECONDS,
        )

        if not response.ok:
            raise ValueError(
            f"Remote API error {response.status_code}: "
            f"{response.text}")

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