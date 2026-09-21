from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
import logging

from app.config import ConfigurationError, validate_configuration
from app.constants import FLOW_BY_EVENT_TYPE
from app.database.notification_repository import get_next_manual_notification_for_user
from app.notifications.models import (
    NotificationRequest,
    NotificationResponse,
    NotificationSendRequest,
)
from app.notifications.sender import NotificationSender
from app.notifications.service import NotificationService
from scripts.apscheduler_runner import shutdown_scheduler, start_scheduler

@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        validate_configuration()
    except ConfigurationError as exc:
        logging.getLogger(__name__).error("Configuration error: %s", exc)
        raise
    start_scheduler()
    try:
        yield
    finally:
        shutdown_scheduler()


app = FastAPI(title="AI-CLAN Notification API", lifespan=lifespan)

service = NotificationService(sender=NotificationSender())


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "AI-CLAN Notification API is running."}




@app.get("/notification/performance")
def get_performance(user_id: int = Query(gt=0)):
    return service.get_performance(user_id)


@app.get("/notification/engagement")
def get_engagement(user_id: int = Query(gt=0)):
    return service.get_engagement(user_id)


@app.get("/notification/sentiment")
def get_sentiment(user_id: int = Query(gt=0)):
    return service.get_sentiment(user_id)

@app.post("/notification/send")
def send_notification(request: NotificationSendRequest):
    try:
        event_type = get_next_manual_notification_for_user(
            request.user_id,
            db_engine=service.db_engine,
        )

        flow = FLOW_BY_EVENT_TYPE.get(event_type)
        if flow is None:
            raise ValueError(f"Unsupported notification event type: {event_type}")

        pipeline_request = NotificationRequest(
            user_id=request.user_id,
            flow=flow,
        )
        result = service.build_notification(pipeline_request)
        if result.remote_send_status == "failed":
            raise HTTPException(status_code=502, detail=result.error or "Remote notification send failed")
        if pipeline_request.should_send and result.remote_send_status == "skipped":
            raise HTTPException(status_code=502, detail=result.error or "Remote notification sender is unavailable")

        notification = NotificationResponse(
            user_id=result.user_id,
            title=result.notification_title,
            description=result.notification_body,
            notification_type=result.notification_type or "VIDEO_RECOMMENDATION",
            reference_id=int(result.reference_id or 0),
            video_popup=result.video_popup,
            image=None,
        )
        notification_data = notification.model_dump()
        return {
            "notification": notification_data,
            "success": not pipeline_request.should_send or result.remote_send_status == "sent",
            "user_id": request.user_id,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
