from fastapi import FastAPI, HTTPException, Query

from app.notifications.models import NotificationRequest
from app.notifications.sender import NotificationSender
from app.notifications.service import NotificationService

app = FastAPI(title="AI-CLAN Notification API")
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