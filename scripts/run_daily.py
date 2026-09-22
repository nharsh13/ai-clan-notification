import logging
import os
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
    test_mode = os.getenv("SCHEDULER_TEST_MODE", "").strip().lower() == "true"
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    total_users = 0
    logger.info("=" * 60)
    logger.info("              AI-CLAN NOTIFICATION SCHEDULER")
    logger.info("=" * 60)
    logger.info("")
    logger.info("[JOB] STARTED")
    logger.info("[JOB] Start Time     : %s", started_at.strftime("%Y-%m-%d %H:%M:%S %Z"))
    if test_mode:
        logger.info("TEST MODE: ENABLED - today's duplicate check is bypassed")
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
                    users = [
                        (row[0], row[1])
                        for row in connection.execute(
                            text(
                                '''
                                SELECT id, COALESCE(NULLIF(name, ''), '') AS user_name
                                FROM public."user"
                                WHERE account_id = 14
                                  AND status = 1
                                  AND debug = false
                                  AND user_type_id = 1
                                '''
                            )
                        )
                    ]

                total_users = len(users)
                logger.info("[JOB] Total Users    : %d", total_users)
                logger.info("-" * 60)
                logger.info("")
                for position, (user_id, user_name) in enumerate(users, start=1):
                    user_label = f"[{position}/{total_users}] USER ID: {user_id} | Name: {user_name or 'Unknown'}"
                    event_type = get_next_notification_for_user(user_id, db_engine=engine)
                    if event_type is None:
                        skipped_count += 1
                        logger.info("%s", user_label)
                        logger.info("        Status            : SKIPPED")
                        logger.info("        Reason            : No eligible notification")
                        continue

                    if not test_mode and has_notification_for_user_on_date(
                        user_id,
                        event_type,
                        db_engine=engine,
                    ):
                        skipped_count += 1
                        logger.info("%s", user_label)
                        logger.info("        Status            : SKIPPED")
                        logger.info("        Reason            : Notification already exists for today")
                        continue

                    flow = FLOW_BY_EVENT_TYPE[event_type]
                    try:
                        result = service.build_notification(NotificationRequest(user_id=user_id, flow=flow))
                        if result is None:
                            skipped_count += 1
                            logger.info("%s", user_label)
                            logger.info("        Status            : SKIPPED")
                            logger.info(
                                "        Reason            : %s",
                                getattr(service, "last_skip_reason", None) or "MISSING_REQUIRED_DATA",
                            )
                            continue
                        if result.remote_send_status == "failed":
                            failed_count += 1
                            logger.error(
                                "%s\n        Notification Type : %s\n        Status            : FAILED\n        Reason            : %s",
                                user_label,
                                result.notification_type,
                                result.error or "unknown send failure",
                            )
                            continue

                        successful_count += 1
                        logger.info("%s", user_label)
                        logger.info("        Notification Type : %s", result.notification_type)
                        logger.info("        Status            : SUCCESS")
                    except Exception as exc:
                        failed_count += 1
                        logger.exception(
                            "%s\n        Status            : FAILED\n        Reason            : %s",
                            user_label,
                            exc,
                        )
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
        logger.info("")
        logger.info("-" * 60)
        logger.info("[JOB] SUMMARY")
        logger.info("-" * 60)
        logger.info("[JOB] Total Users : %d", total_users)
        logger.info("[JOB] Successful  : %d", successful_count)
        logger.info("[JOB] Skipped     : %d", skipped_count)
        logger.info("[JOB] Failed      : %d", failed_count)
        logger.info("-" * 60)
        logger.info("[JOB] COMPLETED")
        logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())