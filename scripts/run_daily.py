import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import text

from app.constants import FLOW_BY_EVENT_TYPE
from app.database.connection import engine
from app.database.notification_repository import (
    get_next_notification_for_user,
    has_notification_for_user_on_date,
)
from app.notifications.models import NotificationRequest
from app.notifications.service import NotificationService

logging.basicConfig(level=logging.INFO)
SCHEDULER_LOCK_KEY = 781234567
SCHEDULER_TIMEZONE = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)

def main() -> None:
    started_at = datetime.now(SCHEDULER_TIMEZONE)
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    logger.info("[JOB] Job STARTED")
    logger.info("[JOB] Start timestamp in IST: %s", started_at.isoformat())
    service = NotificationService()
    try:
        with engine.connect() as lock_connection:
            acquired = lock_connection.execute(
                text("SELECT pg_try_advisory_lock(:lock_key)"),
                {"lock_key": SCHEDULER_LOCK_KEY},
            ).scalar()
            if not acquired:
                skipped_count += 1
                logger.info("[SKIP] Scheduler job skipped: another scheduler is already running")
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

                logger.info("[JOB] Total eligible users fetched: %d", len(user_ids))
                for user_id in user_ids:
                    logger.info("[USER] Current user ID=%s name=unavailable", user_id)
                    logger.info("[USER] Campaign day: not calculated by existing job")
                    event_type = get_next_notification_for_user(user_id, db_engine=engine)
                    if event_type is None:
                        skipped_count += 1
                        logger.info("[SKIP] user_id=%s reason=no eligible notification", user_id)
                        continue

                    if has_notification_for_user_on_date(user_id, event_type, db_engine=engine):
                        skipped_count += 1
                        logger.info("[SKIP] user_id=%s reason=notification already exists for today", user_id)
                        continue

                    flow = FLOW_BY_EVENT_TYPE[event_type]
                    try:
                        result = service.build_notification(NotificationRequest(user_id=user_id, flow=flow))
                        if result is None:
                            skipped_count += 1
                            logger.info("[SKIP] user_id=%s reason=no notification generated", user_id)
                            continue
                        if result.remote_send_status == "failed":
                            failed_count += 1
                            logger.error(
                                "[ERROR] Failed notification user_id=%s flow=%s error=%s",
                                user_id,
                                flow,
                                result.error or "unknown send failure",
                            )
                            continue

                        processed_count += 1
                        logger.info("[SUCCESS] Notification processed successfully user_id=%s flow=%s", user_id, flow)
                    except Exception as exc:
                        failed_count += 1
                        logger.exception("[ERROR] Failed notification user_id=%s flow=%s error=%s", user_id, flow, exc)
            finally:
                lock_connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": SCHEDULER_LOCK_KEY},
                )
    except Exception:
        logger.exception("[ERROR] Unexpected job exception")
        raise
    finally:
        ended_at = datetime.now(SCHEDULER_TIMEZONE)
        logger.info("[JOB] Job ENDED")
        logger.info("[JOB] End timestamp in IST: %s", ended_at.isoformat())
        logger.info("[JOB] Total execution duration: %s", ended_at - started_at)
        logger.info(
            "[JOB] Overall completion status: %s (successful=%d skipped=%d failed=%d)",
            "COMPLETED" if failed_count == 0 else "COMPLETED_WITH_ERRORS",
            processed_count,
            skipped_count,
            failed_count,
        )


if __name__ == "__main__":
    sys.exit(main())