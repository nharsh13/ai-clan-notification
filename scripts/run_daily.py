import asyncio
import logging
import os
import sys
import time
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
MAX_CONCURRENT_USERS = 5
logger = logging.getLogger(__name__)


def _collect_users() -> list[tuple[int, str]]:
    with engine.connect() as connection:
        return [
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


async def _process_user_async(
    user_id: int,
    user_name: str,
    *,
    position: int,
    total_users: int,
    test_mode: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    user_label = f"[{position}/{total_users}] USER ID: {user_id} | Name: {user_name or 'Unknown'}"
    async with semaphore:
        try:
            event_type = get_next_notification_for_user(user_id, db_engine=engine)
            if event_type is None:
                logger.info("%s", user_label)
                logger.info("        Status            : SKIPPED")
                logger.info("        Reason            : No eligible notification")
                return {"status": "skipped", "user_id": user_id, "reason": "No eligible notification"}

            if not test_mode and has_notification_for_user_on_date(
                user_id,
                event_type,
                db_engine=engine,
            ):
                logger.info("%s", user_label)
                logger.info("        Status            : SKIPPED")
                logger.info("        Reason            : Notification already exists for today")
                return {"status": "skipped", "user_id": user_id, "reason": "Notification already exists for today"}

            flow = FLOW_BY_EVENT_TYPE[event_type]
            service = NotificationService()
            processor = getattr(service, "build_notification_async", None)
            if processor is not None:
                result = await processor(NotificationRequest(user_id=user_id, flow=flow))
            else:
                result = await asyncio.to_thread(service.build_notification, NotificationRequest(user_id=user_id, flow=flow))
            if result is None:
                logger.info("%s", user_label)
                logger.info("        Status            : SKIPPED")
                logger.info(
                    "        Reason            : %s",
                    getattr(service, "last_skip_reason", None) or "MISSING_REQUIRED_DATA",
                )
                return {"status": "skipped", "user_id": user_id, "reason": getattr(service, "last_skip_reason", None) or "MISSING_REQUIRED_DATA"}
            if result.remote_send_status == "failed":
                logger.error(
                    "%s\n        Notification Type : %s\n        Status            : FAILED\n        Reason            : %s",
                    user_label,
                    result.notification_type,
                    result.error or "unknown send failure",
                )
                return {"status": "failed", "user_id": user_id, "reason": result.error or "unknown send failure"}

            timings = getattr(service, "timings", {})
            logger.info("%s", user_label)
            logger.info("        Notification Type : %s", result.notification_type)
            logger.info("        DB Context        : %.2fs", timings.get("DB Context", 0.0))
            if "Recommendation" in timings:
                logger.info("        Recommendation    : %.2fs", timings["Recommendation"])
            if "LLM" in timings:
                logger.info("        LLM               : %.2fs", timings["LLM"])
            if "Remote API" in timings:
                logger.info("        Remote API        : %.2fs", timings["Remote API"])
            total_elapsed = timings.get("Total", 0.0)
            if not total_elapsed:
                total_elapsed = sum(timings.values())
            logger.info("        Total             : %.2fs", total_elapsed)
            logger.info("        Status            : SUCCESS")
            return {"status": "success", "user_id": user_id, "notification_type": result.notification_type}
        except Exception as exc:
            logger.exception(
                "%s\n        Status            : FAILED\n        Reason            : %s",
                user_label,
                exc,
            )
            return {"status": "failed", "user_id": user_id, "reason": str(exc)}


async def run_scheduler_async(test_mode: bool | None = None) -> dict:
    test_mode = bool(test_mode) if test_mode is not None else os.getenv("SCHEDULER_TEST_MODE", "").strip().lower() == "true"
    users = _collect_users()
    total_users = len(users)
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_USERS)
    tasks = [
        asyncio.create_task(
            _process_user_async(
                user_id=user_id,
                user_name=user_name,
                position=position,
                total_users=total_users,
                test_mode=test_mode,
                semaphore=semaphore,
            )
        )
        for position, (user_id, user_name) in enumerate(users, start=1)
    ]
    results = await asyncio.gather(*tasks)
    for result in results:
        if isinstance(result, dict):
            if result.get("status") == "success":
                successful_count += 1
            elif result.get("status") == "failed":
                failed_count += 1
            elif result.get("status") == "skipped":
                skipped_count += 1
        else:
            successful_count += 1
    return {
        "total_users": total_users,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
    }


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
                summary = asyncio.run(run_scheduler_async(test_mode=test_mode))
                total_users = summary["total_users"]
                successful_count = summary["successful_count"]
                failed_count = summary["failed_count"]
                skipped_count = summary["skipped_count"]
                logger.info("[JOB] Total Users    : %d", total_users)
                logger.info("-" * 60)
                logger.info("")
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