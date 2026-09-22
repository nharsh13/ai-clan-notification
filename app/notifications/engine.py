
from app.notifications.models import Notification


class NotificationEngine:
    """Build AI-CLAN notification payloads."""

    def __init__(self):
        self.config = {}

    def make_notification(
        self,
        *,
        user_id: int,
        notification_type: str,
        title: str,
        description: str | None = None,
        reference_id: int = 0,
        video_popup: bool | None = None,
        image: str | None = None,
    ) -> Notification:
        """
        Build a notification using the current AI-CLAN notification contract.

        Parameters:
            user_id:
                ID of the user receiving the notification.

            notification_type:
                Type of notification, for example:
                PERFORMANCE, SENTIMENT_QA, SENTIMENT_ENGAGEMENT.

            title:
                Notification title.

            description:
                Notification description/body.

            reference_id:
                ID related to the notification.
                Defaults to 0 when there is no reference.

            video_popup:
                Whether the notification should open/show a video popup.

            image:
                Optional notification image URL/path.

        Returns:
            Notification:
                A validated notification payload.
        """

        return Notification(
            user_id=user_id,
            title=title,
            description=description,
            notification_type=notification_type,
            reference_id=reference_id,
            video_popup=video_popup,
            image=image,
        )
