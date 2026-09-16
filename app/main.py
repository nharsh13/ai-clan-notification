from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
import logging

from app.config import ConfigurationError, validate_configuration
from app.notifications.models import NotificationRequest, NotificationSendRequest
from app.notifications.sender import NotificationSender
from app.notifications.service import NotificationService

@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        validate_configuration()
    except ConfigurationError as exc:
        logging.getLogger(__name__).error("Configuration error: %s", exc)
        raise
    yield


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
        pipeline_request = NotificationRequest(
            user_id=request.user_id,
            campaign_day=request.campaign_day,
            flow="performance",
        )
        result = service.build_notification(pipeline_request)
        if result.remote_send_status == "failed":
            raise HTTPException(status_code=502, detail=result.error or "Remote notification send failed")
        if pipeline_request.should_send and result.remote_send_status == "skipped":
            raise HTTPException(status_code=502, detail=result.error or "Remote notification sender is unavailable")

        notification = {
            "action": result.action,
            "audience_strategy": result.audience_strategy,
            "campaign_day": result.campaign_day,
            "cohort_key": result.cohort_key,
            "creator_name": result.creator_name,
            "deep_link": result.deep_link,
            "notification_body": result.notification_body,
            "notification_title": result.notification_title,
            "notification_type": result.notification_type,
            "reference_id": result.reference_id,
            "should_send": result.should_send,
            "video_id": result.video_id,
            "video_popup": result.video_popup,
            "video_title": result.video_title,
        }
        return {
            "notification": notification,
            "remote_send_response": result.remote_send_response,
            "remote_send_status": result.remote_send_status or "skipped",
            "success": not pipeline_request.should_send or result.remote_send_status == "sent",
            "user_id": request.user_id,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
