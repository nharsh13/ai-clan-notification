import asyncio
import inspect
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

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)
root_handler = logging.StreamHandler()
root_handler.setFormatter(logging.Formatter("%(message)s"))
root_logger.addHandler(root_handler)

for noisy_logger_name in (
    "sqlalchemy",
    "sqlalchemy.engine",
    "sqlalchemy.pool",
    "urllib3",
    "requests",
    "httpx",
    "app.notifications.sender",
    "app.llm.llm_client",
    "app.llm.notification_generator",
    "app.notifications.service",
    "app.recommendation.recommendation",
    "openai._base_client",
):
    logging.getLogger(noisy_logger_name).setLevel(logging.CRITICAL)

SCHEDULER_LOCK_KEY = 781234567
SCHEDULER_TIMEZONE = ZoneInfo("Asia/Kolkata")
MAX_CONCURRENT_USERS = 5
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def configure_terminal_logging() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    for noisy_logger_name in (
        "huggingface_hub",
        "transformers",
        "sentence_transformers",
        "tokenizers",
        "torch",
        "urllib3",
        "urllib3.connectionpool",
        "urllib3.util.retry",
        "urllib3.util",
        "requests",
        "requests.packages.urllib3",
        "requests.packages.urllib3.connectionpool",
        "httpx",
        "httpcore",
        "openai._base_client",
    ):
        logging.getLogger(noisy_logger_name).setLevel(logging.ERROR)
        logging.getLogger(noisy_logger_name).propagate = False

    try:
        from huggingface_hub import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except Exception:
        pass


configure_terminal_logging()


def _log_user_status(
    *,
    position: int,
    total_users: int,
    user_id: int,
    user_name: str,
    notification_type: str,
    status: str,
    http_status: int | None = None,
) -> None:
    if status == "STARTED":
        logger.info("")
        logger.info("[%s/%s] USER %s | %s", position, total_users or position, user_id, user_name)
        logger.info("")
        logger.info("        TYPE   : %s", notification_type)
        logger.info("        STATUS : STARTED")
        return

    remote_label = "N/A" if http_status is None else http_status
    if status == "SUCCESS":
        remote_status = "SUCCESS"
    elif status == "FAILED":
        remote_status = "FAILED"
    elif status == "SKIPPED":
        remote_status = "SKIPPED"
    else:
        remote_status = "FAILED"
    logger.info("")
    logger.info("[%s/%s] USER %s | %s", position, total_users or position, user_id, user_name)
    logger.info("")
    logger.info("        TYPE   : %s", notification_type)
    logger.info("        STATUS : %s", status)
    logger.info("")
    logger.info("[REMOTE] USER %s | HTTP %s | %s", user_id, remote_label, remote_status)


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
    completion_counter: dict[str, int] | None = None,
    completion_counter_lock: asyncio.Lock | None = None,
) -> dict:
    async def _mark_completion() -> int:
        if completion_counter is not None and completion_counter_lock is not None:
            async with completion_counter_lock:
                completion_counter["value"] += 1
                return completion_counter["value"]
        return position

    async with semaphore:
        try:
            event_type = get_next_notification_for_user(user_id, db_engine=engine)
            if event_type is None:
                completion_index = await _mark_completion()
                _log_user_status(position=position, total_users=total_users, user_id=user_id,
                                 user_name=user_name or "Unknown", notification_type="UNKNOWN",
                                 status="SKIPPED")
                return {
                    "status": "skipped",
                    "user_id": user_id,
                    "user_name": user_name or "Unknown",
                    "notification_type": "UNKNOWN",
                    "http_status": None,
                    "completion_index": completion_index,
                    "total_users": total_users,
                    "reason": "No eligible notification",
                }

            if not test_mode and has_notification_for_user_on_date(
                user_id,
                event_type,
                db_engine=engine,
            ):
                completion_index = await _mark_completion()
                _log_user_status(position=position, total_users=total_users, user_id=user_id,
                                 user_name=user_name or "Unknown", notification_type=event_type,
                                 status="SKIPPED")
                return {
                    "status": "skipped",
                    "user_id": user_id,
                    "user_name": user_name or "Unknown",
                    "notification_type": event_type,
                    "http_status": None,
                    "completion_index": completion_index,
                    "total_users": total_users,
                    "reason": "Notification already exists for today",
                }

            _log_user_status(
                position=position,
                total_users=total_users,
                user_id=user_id,
                user_name=user_name or "Unknown",
                notification_type=event_type,
                status="STARTED",
            )

            flow = FLOW_BY_EVENT_TYPE[event_type]
            service = NotificationService()
            processor = getattr(service, "build_notification_async", None)
            try:
                if processor is not None:
                    result = await processor(NotificationRequest(user_id=user_id, flow=flow))
                else:
                    result = await asyncio.to_thread(service.build_notification, NotificationRequest(user_id=user_id, flow=flow))
            except TypeError:
                if processor is not None:
                    result = await processor(NotificationRequest(user_id=user_id, flow=flow))
                else:
                    result = await asyncio.to_thread(service.build_notification, NotificationRequest(user_id=user_id, flow=flow))
            if result is None:
                completion_index = await _mark_completion()
                final_result = {
                    "status": "skipped",
                    "user_id": user_id,
                    "user_name": user_name or "Unknown",
                    "notification_type": event_type,
                    "http_status": None,
                    "completion_index": completion_index,
                    "total_users": total_users,
                    "reason": getattr(service, "last_skip_reason", None) or "MISSING_REQUIRED_DATA",
                }
                _log_user_status(
                    position=position,
                    total_users=total_users,
                    user_id=user_id,
                    user_name=user_name or "Unknown",
                    notification_type=event_type,
                    status="SKIPPED",
                )
                return final_result
            reason = getattr(result, "reason", None) or getattr(service, "last_skip_reason", None)
            remote_status = getattr(result, "remote_send_status", None)
            http_status = None
            if getattr(result, "remote_send_response", None):
                http_status = result.remote_send_response.get("status_code")
            # A remote send is successful only when the API explicitly returns HTTP 200.
            # Notification-building skips remain business skips; ambiguous outcomes fail closed.
            business_skip = reason in {
                "NO_ENGAGEMENT_DATA", "NO_QA_DATA", "NO_VIDEO_RECOMMENDATION",
                "MISSING_REQUIRED_DATA", "LLM_NO_RESPONSE",
            }
            if (
                remote_status == "failed"
                or (remote_status == "sent" and http_status != 200)
                or remote_status not in {"sent", None}
                or (remote_status is None and not business_skip)
            ):
                completion_index = await _mark_completion()
                final_result = {
                    "status": "failed",
                    "user_id": user_id,
                    "user_name": user_name or "Unknown",
                    "notification_type": getattr(result, "notification_type", event_type),
                    "http_status": http_status,
                    "completion_index": completion_index,
                    "total_users": total_users,
                    "reason": getattr(result, "error", None) or reason or "unknown send failure",
                }
                _log_user_status(
                    position=position,
                    total_users=total_users,
                    user_id=user_id,
                    user_name=user_name or "Unknown",
                    notification_type=getattr(result, "notification_type", event_type),
                    status="FAILED",
                    http_status=http_status,
                )
                return final_result
            if remote_status is None and business_skip:
                completion_index = await _mark_completion()
                final_result = {
                    "status": "skipped",
                    "user_id": user_id,
                    "user_name": user_name or "Unknown",
                    "notification_type": getattr(result, "notification_type", event_type),
                    "http_status": http_status,
                    "completion_index": completion_index,
                    "total_users": total_users,
                    "reason": reason or getattr(result, "error", None) or "MISSING_REQUIRED_DATA",
                }
                _log_user_status(
                    position=position,
                    total_users=total_users,
                    user_id=user_id,
                    user_name=user_name or "Unknown",
                    notification_type=getattr(result, "notification_type", event_type),
                    status="SKIPPED",
                    http_status=http_status,
                )
                return final_result

            completion_index = await _mark_completion()
            final_result = {
                "status": "success",
                "user_id": user_id,
                "user_name": user_name or "Unknown",
                "notification_type": getattr(result, "notification_type", event_type),
                "http_status": http_status,
                "completion_index": completion_index,
                "total_users": total_users,
                "reason": None,
            }
            _log_user_status(
                position=position,
                total_users=total_users,
                user_id=user_id,
                user_name=user_name or "Unknown",
                notification_type=getattr(result, "notification_type", event_type),
                status="SUCCESS",
                http_status=http_status,
            )
            return final_result
        except Exception as exc:
            completion_index = await _mark_completion()
            _log_user_status(position=position, total_users=total_users, user_id=user_id,
                             user_name=user_name or "Unknown", notification_type="UNKNOWN",
                             status="FAILED")
            return {
                "status": "failed",
                "user_id": user_id,
                "user_name": user_name or "Unknown",
                "notification_type": "UNKNOWN",
                "http_status": None,
                "completion_index": completion_index,
                "total_users": total_users,
                "reason": str(exc),
            }


async def run_scheduler_async(test_mode: bool | None = None) -> dict:
    test_mode = bool(test_mode) if test_mode is not None else os.getenv("SCHEDULER_TEST_MODE", "").strip().lower() == "true"
    users = _collect_users()
    total_users = len(users)
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_USERS)
    completion_counter = {"value": 0}
    completion_lock = asyncio.Lock()
    task_kwargs = {
        "user_id": None,
        "user_name": None,
        "position": None,
        "total_users": total_users,
        "test_mode": test_mode,
        "semaphore": semaphore,
    }
    supports_completion_tracking = "completion_counter" in inspect.signature(_process_user_async).parameters
    if supports_completion_tracking:
        task_kwargs["completion_counter"] = completion_counter
        task_kwargs["completion_counter_lock"] = completion_lock

    tasks = []
    for position, (user_id, user_name) in enumerate(users, start=1):
        item_kwargs = dict(task_kwargs)
        item_kwargs["user_id"] = user_id
        item_kwargs["user_name"] = user_name
        item_kwargs["position"] = position
        tasks.append(asyncio.create_task(_process_user_async(**item_kwargs)))
    results = await asyncio.gather(*tasks)
    details = []
    for result in results:
        if not isinstance(result, dict):
            continue
        details.append(result)
        if result.get("status") == "success":
            successful_count += 1
        elif result.get("status") == "failed":
            failed_count += 1
        elif result.get("status") == "skipped":
            skipped_count += 1
    return {
        "total_users": total_users,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "details": details,
    }


def main() -> None:
    started_at = datetime.now(SCHEDULER_TIMEZONE)
    test_mode = os.getenv("SCHEDULER_TEST_MODE", "").strip().lower() == "true"
    cron_hour = os.getenv("CRON_HOUR", "00")
    cron_minute = os.getenv("CRON_MINUTE", "00")

    logger.info("[SCHEDULER] Scheduler status: STARTED")
    logger.info("[SCHEDULER] Current date/time in IST: %s", started_at.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("[SCHEDULER] Configured schedule: cron[hour='%s', minute='%s']", cron_hour, cron_minute)
    logger.info("[SCHEDULER] Status: RUNNING")
    logger.info("")
    logger.info("[JOB] Notification job started")
    if test_mode:
        logger.info("[JOB] TEST MODE: ENABLED")
    logger.info("")

    try:
        with engine.connect() as lock_connection:
            acquired = lock_connection.execute(
                text("SELECT pg_try_advisory_lock(:lock_key)"),
                {"lock_key": SCHEDULER_LOCK_KEY},
            ).scalar()
            if not acquired:
                logger.info("[JOB] SUMMARY")
                logger.info("[JOB] Total Users : 0")
                logger.info("[JOB] Successful  : 0")
                logger.info("[JOB] Failed      : 0")
                logger.info("[JOB] Skipped     : 1")
                logger.info("")
                logger.info("[JOB] ALL NOTIFICATIONS SENT")
                logger.info("[JOB] COMPLETED")
                return

            try:
                summary = asyncio.run(run_scheduler_async(test_mode=test_mode))
                details = summary.get("details", [])
                total_users = summary["total_users"]
                successful_count = summary["successful_count"]
                failed_count = summary["failed_count"]
                skipped_count = summary["skipped_count"]

                logger.info("")
                logger.info("[JOB] SUMMARY")
                logger.info("[JOB] Total Users : %d", total_users)
                logger.info("[JOB] Successful  : %d", successful_count)
                logger.info("[JOB] Failed      : %d", failed_count)
                logger.info("[JOB] Skipped     : %d", skipped_count)
                logger.info("")
                logger.info("[JOB] ALL NOTIFICATIONS SENT")
                logger.info("[JOB] COMPLETED")
            finally:
                lock_connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": SCHEDULER_LOCK_KEY},
                )
    except Exception:
        logger.exception("[ERROR] Unexpected job exception")
        raise


if __name__ == "__main__":
    sys.exit(main())
