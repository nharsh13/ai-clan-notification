from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
import logging

from app.config import ConfigurationError, validate_configuration
from app.notifications.models import NotificationRequest
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


@app.post("/notification/send")
def send_notification(request: NotificationRequest):
    try:
        if request.flow == "sentiment":
            return service.process_sentiment_notification(request.user_id)
        return service.build_notification(request).model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/notification/performance")
def get_performance(user_id: int = Query(gt=0)):
    return service.get_performance(user_id)


@app.get("/notification/engagement")
def get_engagement(user_id: int = Query(gt=0)):
    return service.get_engagement(user_id)


@app.get("/notification/sentiment")
def get_sentiment(user_id: int = Query(gt=0)):
    return service.get_sentiment(user_id)