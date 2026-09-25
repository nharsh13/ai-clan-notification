import asyncio
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

SCHEDULER_TIMEZONE = ZoneInfo("Asia/Kolkata")
MAX_CONCURRENT_USERS = 5
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_test_mode(value: object | None) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


def get_scheduler_test_mode() -> bool:
    return parse_test_mode(os.getenv("SCHEDULER_TEST_MODE", "").strip())


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
    user_id: int,
    user_name: str,
    notification_type: str,
    status: str,
    http_status: int | None = None,
    reason: str | None = None,
    raw_response: str | None = None,
    terminal_message: str | None = None,
) -> None:
    name = user_name or "Unknown"
    notification_type = notification_type or "UNKNOWN"
    if reason:
        reason = " ".join(str(reason).split())
    prefix = f"{user_id} | {name:<30} | {notification_type:<22} | {status:<7}"
    if status == "SUCCESS":
        logger.info("%s %s", user_id, raw_response or "")
    elif terminal_message:
        logger.info("%s %s", user_id, terminal_message)
    elif status == "FAILED":
        http = f"HTTP {http_status}" if http_status is not None else "HTTP unavailable"
        logger.info("%s | %s | %s", prefix, http, reason or "Unknown error")
    elif status == "SKIPPED":
        logger.info("%s | %s", prefix, reason or "Unspecified")


def _classify_terminal_error(error: object, error_type: str | None = None) -> str | None:
    message = str(error or "").lower()
    if any(marker in message for marker in (
        "insufficient_quota", "quota exceeded", "billing", "credit exhausted",
        "out of credits", "exceeded your current quota", "usage limit",
    )):
        return "LLM_CREDIT_OUT"
    exception_type = (error_type or type(error).__name__).lower()
    if any(marker in exception_type for marker in (
        "connecterror", "connecttimeout", "readtimeout", "timeout", "connectionerror",
    )) or any(marker in message for marker in (
        "connection refused", "connection error", "connecterror", "connecttimeout",
        "readtimeout", "network is unreachable", "name or service not known",
        "temporary failure in name resolution", "failed to establish a new connection",
        "all connection attempts failed", "remote end closed connection",
    )):
        return "REMOTE_URL_NOT_RESPONDING"
    return None


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
    test_mode: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        try:
            event_type = get_next_notification_for_user(user_id, db_engine=engine)
            if event_type is None:
                _log_user_status(user_id=user_id,
                                 user_name=user_name or "Unknown", notification_type="UNKNOWN",
                                 status="SKIPPED", reason="NO_ELIGIBLE_NOTIFICATION")
                return {
                    "status": "skipped",
                    "user_id": user_id,
                    "user_name": user_name or "Unknown",
                    "notification_type": "UNKNOWN",
                    "http_status": None,
                    "reason": "No eligible notification",
                }

            if not test_mode and has_notification_for_user_on_date(
                user_id,
                event_type,
                db_engine=engine,
            ):
                _log_user_status(user_id=user_id,
                                 user_name=user_name or "Unknown", notification_type=event_type,
                                 status="SKIPPED", reason="ALREADY_NOTIFIED_TODAY")
                return {
                    "status": "skipped",
                    "user_id": user_id,
                    "user_name": user_name or "Unknown",
                    "notification_type": event_type,
                    "http_status": None,
                    "reason": "Notification already exists for today",
                }

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
                skip_reason = getattr(service, "last_skip_reason", None) or "MISSING_REQUIRED_DATA"
                classified_error = _classify_terminal_error(
                    getattr(service, "last_error", None), getattr(service, "last_error_type", None)
                )
                final_status = "failed" if skip_reason == "LLM_NO_RESPONSE" else "skipped"
                final_result = {
                    "status": final_status,
                    "user_id": user_id,
                    "user_name": user_name or "Unknown",
                    "notification_type": event_type,
                    "http_status": None,
                    "reason": getattr(service, "last_error", None) or skip_reason,
                }
                _log_user_status(
                    user_id=user_id,
                    user_name=user_name or "Unknown",
                    notification_type=event_type,
                    status=final_status.upper(),
                    reason=final_result["reason"],
                    terminal_message=classified_error,
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
                failure_reason = getattr(result, "error", None) or reason or "unknown send failure"
                terminal_message = _classify_terminal_error(
                    failure_reason, getattr(result, "remote_send_error_type", None)
                )
                final_result = {
                    "status": "failed",
                    "user_id": user_id,
                    "user_name": user_name or "Unknown",
                    "notification_type": getattr(result, "notification_type", event_type),
                    "http_status": http_status,
                    "reason": failure_reason,
                }
                _log_user_status(
                    user_id=user_id,
                    user_name=user_name or "Unknown",
                    notification_type=getattr(result, "notification_type", event_type),
                    status="FAILED",
                    http_status=http_status,
                    reason=failure_reason,
                    terminal_message=terminal_message,
                )
                return final_result
            if remote_status is None and business_skip:
                final_result = {
                    "status": "skipped",
                    "user_id": user_id,
                    "user_name": user_name or "Unknown",
                    "notification_type": getattr(result, "notification_type", event_type),
                    "http_status": http_status,
                    "reason": reason or getattr(result, "error", None) or "MISSING_REQUIRED_DATA",
                }
                _log_user_status(
                    user_id=user_id,
                    user_name=user_name or "Unknown",
                    notification_type=getattr(result, "notification_type", event_type),
                    status="SKIPPED",
                    http_status=http_status,
                    reason=reason or getattr(result, "error", None) or "MISSING_REQUIRED_DATA",
                )
                return final_result

            final_result = {
                "status": "success",
                "user_id": user_id,
                "user_name": user_name or "Unknown",
                "notification_type": getattr(result, "notification_type", event_type),
                "http_status": http_status,
                "reason": None,
            }
            _log_user_status(
                user_id=user_id,
                user_name=user_name or "Unknown",
                notification_type=getattr(result, "notification_type", event_type),
                status="SUCCESS",
                http_status=http_status,
                raw_response=(getattr(result, "remote_send_response", None) or {}).get("raw_response", ""),
            )
            return final_result
        except Exception as exc:
            terminal_message = _classify_terminal_error(exc)
            _log_user_status(user_id=user_id,
                             user_name=user_name or "Unknown", notification_type="UNKNOWN",
                             status="FAILED", reason=str(exc), terminal_message=terminal_message)
            return {
                "status": "failed",
                "user_id": user_id,
                "user_name": user_name or "Unknown",
                "notification_type": "UNKNOWN",
                "http_status": None,
                "reason": str(exc),
            }


async def run_scheduler_async(test_mode: bool | None = None) -> dict:
    started = datetime.now(SCHEDULER_TIMEZONE)
    test_mode = parse_test_mode(test_mode) if test_mode is not None else get_scheduler_test_mode()
    users = _collect_users()
    total_users = len(users)
    logger.info("============================================================")
    logger.info("                 AI-CLAN NOTIFICATION JOB")
    logger.info("============================================================")
    logger.info("Started             : %s IST", started.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Mode                : %s", "TEST" if test_mode else "PRODUCTION")
    logger.info("SCHEDULER_TEST_MODE : %s (%s)", "TRUE" if test_mode else "FALSE", "ENABLED" if test_mode else "DISABLED")
    logger.info("Total Users         : %d", total_users)
    logger.info("============================================================")
    logger.info("")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_USERS)
    tasks = [
        asyncio.create_task(
            _process_user_async(
                user_id,
                user_name,
                test_mode=test_mode,
                semaphore=semaphore,
            )
        )
        for user_id, user_name in users
    ]
    results = await asyncio.gather(*tasks)
    return {
        "total_users": total_users,
        "details": results,
    }


def main() -> None:
    test_mode = get_scheduler_test_mode()

    try:
        asyncio.run(run_scheduler_async(test_mode=test_mode))
        logger.info("============================================================")
        logger.info("                 AI-CLAN NOTIFICATION SENT")
        logger.info("============================================================")
    except Exception:
        logger.exception("[ERROR] Unexpected job exception")
        raise


if __name__ == "__main__":
    sys.exit(main())
