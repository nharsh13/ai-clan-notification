from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
import logging

from app.config import ConfigurationError, validate_configuration
from app.notifications.models import (
    NotificationResponse,
    NotificationSendRequest,
)
from app.notifications.sender import NotificationSender
from app.notifications.service import NotificationService
from scripts.apscheduler_runner import shutdown_scheduler, start_scheduler

for noisy_logger_name in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

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
        result = service.build_notification(request.user_id)
        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Notification could not be generated",
            )
        if result.remote_send_status == "failed":
            raise HTTPException(status_code=502, detail=result.error or "Remote notification send failed")
        if result.should_send and result.remote_send_status == "skipped":
            raise HTTPException(status_code=502, detail=result.error or "Remote notification sender is unavailable")

        notification = NotificationResponse(
            user_id=result.user_id,
            title=result.title,
            description=result.description,
            notification_type=result.notification_type,
            reference_id=result.reference_id,
            video_popup=result.video_popup,
            image=result.image,
        )
        notification_data = notification.model_dump()
        return {
            "notification": notification_data,
            "success": not result.should_send or result.remote_send_status == "sent",
            "user_id": request.user_id,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
