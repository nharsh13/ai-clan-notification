from app.notifications.models import FlowName, Notification


class NotificationEngine:
    """Build AI-CLAN notification payloads without the 30-Day campaign YAML dependency."""

    def __init__(self):
        self.config = {}

    @staticmethod
    def _normalize_flow(flow: str) -> FlowName:
        normalized = str(flow).strip().lower()
        if normalized in {"performance", "engagement", "sentiment"}:
            return normalized  # type: ignore[return-value]
        return "performance"

    def make_notification(
        self,
        *,
        user_id: int,
        flow: str,
        notification_type: str,
        title: str,
        description: str,
        reference_id: int | None = None,
        deep_link: str | None = None,
        video_id: int | None = None,
        video_title: str | None = None,
        creator_name: str | None = None,
        should_send: bool = True,
        video_popup: str | None = None,
    ) -> Notification:
        normalized_flow = self._normalize_flow(flow)
        return Notification(
            user_id=user_id,
            flow=normalized_flow,
            notification_title=title,
            notification_body=description,
            notification_type=notification_type,
            should_send=should_send,
            reference_id=reference_id,
            deep_link=deep_link,
            video_id=video_id,
            video_title=video_title,
            creator_name=creator_name,
            video_popup=video_popup,
            audience_strategy="dynamic",
            cohort_key="ai_clan",
            action="send",
        )