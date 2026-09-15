import logging
import sys

from sqlalchemy import text

from app.database.connection import engine
from app.notifications.models import NotificationRequest
from app.notifications.service import NotificationService

logging.basicConfig(level=logging.INFO)


def main() -> None:
    service = NotificationService()
    with engine.connect() as connection:
        user_ids = [row[0] for row in connection.execute(text('SELECT id FROM public."user" WHERE id IS NOT NULL'))]
    for user_id in user_ids:
        for flow in ("performance", "engagement", "sentiment"):
            result = service.build_notification(NotificationRequest(user_id=user_id, flow=flow))
            if not result.success:
                logging.getLogger(__name__).error("Notification failed for user=%s flow=%s", user_id, flow)


if __name__ == "__main__":
    sys.exit(main())