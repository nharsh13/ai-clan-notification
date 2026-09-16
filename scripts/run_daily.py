import logging
import sys
from datetime import datetime, timezone

from sqlalchemy import text

from app.database.connection import engine
from app.database.notification_repository import (
    get_next_notification_for_user,
    has_notification_for_user_on_date,
    insert_notification,
)
from app.notifications.models import NotificationRequest
from app.notifications.service import NotificationService

logging.basicConfig(level=logging.INFO)

FLOW_BY_EVENT_TYPE = {
    "VIDEO_RECOMMENDATION": "performance",
    "SENTIMENT_ENGAGEMENT": "engagement",
    "SENTIMENT_QA": "sentiment",
}


def main() -> None:
    service = NotificationService()
    with engine.connect() as connection:
        user_ids = [row[0] for row in connection.execute(text('SELECT id FROM public."user" WHERE id IS NOT NULL'))]

    for user_id in user_ids:
        event_type = get_next_notification_for_user(user_id, db_engine=engine)
        if event_type is None:
            continue

        if has_notification_for_user_on_date(user_id, event_type, db_engine=engine):
            continue

        flow = FLOW_BY_EVENT_TYPE[event_type]
        try:
            result = service.build_notification(NotificationRequest(user_id=user_id, flow=flow))
            if result.remote_send_status == "failed":
                logging.getLogger(__name__).error("Notification failed for user=%s flow=%s", user_id, flow)
                continue

            insert_notification(
                target_user_id=user_id,
                event_type=event_type,
                title=result.notification_title,
                description=result.notification_body,
                event_ref_id=result.reference_id,
                event_details=result.model_dump(exclude_none=True),
                created_at=datetime.now(timezone.utc),
                db_engine=engine,
            )
        except Exception:
            logging.getLogger(__name__).exception("Notification failed for user=%s flow=%s", user_id, flow)


if __name__ == "__main__":
    sys.exit(main())