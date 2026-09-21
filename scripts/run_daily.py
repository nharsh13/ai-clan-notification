import logging
import sys
from datetime import datetime, timezone

from sqlalchemy import text

from app.constants import FLOW_BY_EVENT_TYPE
from app.database.connection import engine
from app.database.notification_repository import (
    get_next_notification_for_user,
    has_notification_for_user_on_date,
    insert_notification,
)
from app.notifications.models import NotificationRequest
from app.notifications.service import NotificationService

logging.basicConfig(level=logging.INFO)
SCHEDULER_LOCK_KEY = 781234567

def main() -> None:
    service = NotificationService()
    with engine.connect() as lock_connection:
        acquired = lock_connection.execute(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": SCHEDULER_LOCK_KEY},
        ).scalar()
        if not acquired:
            logging.getLogger(__name__).info("Another scheduler is already running; exiting")
            return

        try:
            with engine.connect() as connection:
                user_ids = [
                    row[0]
                    for row in connection.execute(
                        text(
                            '''
                            SELECT id
                            FROM public."user"
                            WHERE account_id = 14
                              AND status = 1
                              AND debug = false
                              AND user_type_id = 1
                            '''
                        )
                    )
                ]

            for user_id in user_ids:
                event_type = get_next_notification_for_user(user_id, db_engine=engine)
                if event_type is None:
                    continue

                if has_notification_for_user_on_date(user_id, event_type, db_engine=engine):
                    continue

                flow = FLOW_BY_EVENT_TYPE[event_type]
                try:
                    result = service.build_notification(NotificationRequest(user_id=user_id, flow=flow))
                    if result is None:
                        continue
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
        finally:
            lock_connection.execute(
                text("SELECT pg_advisory_unlock(:lock_key)"),
                {"lock_key": SCHEDULER_LOCK_KEY},
            )


if __name__ == "__main__":
    sys.exit(main())