import app.notifications.service as service_module
from app.notifications.models import NotificationRequest


class DummySender:
    def __init__(self):
        self.remote_url = "https://example.test/notify"

    def send(self, **kwargs):
        return {"status": "ok", "payload": kwargs}


class DummyGenerator:
    def generate(self, prompt):
        return {
            "title": "Growth Check",
            "description": "Keep working on the area that needs the most attention.",
        }


def test_build_notification_performance_flow(monkeypatch):
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "performance": [{"kii_id": 117, "kii_name": "KII 117", "daily_target": 2.0, "seven_day_target": 14.0, "seven_day_actual": 10.0, "performance_percentage": 71.43}],
        "improvement_area": {"kii_id": 117, "kii_name": "KII 117", "performance_percentage": 71.43},
    })
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "language_code": "en",
        "video_language_id": 1,
    })
    monkeypatch.setattr(service_module, "recommend_video", lambda *args, **kwargs: {"video_id": 55, "title": "Focus on clarity"})

    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())
    result = service.build_notification(NotificationRequest(user_id=953, flow="performance", should_send=False))

    assert result.user_id == 953
    assert result.flow == "performance"
    assert result.notification_title == "Growth Check"
    assert result.video_id == 55
    assert result.video_title == "Focus on clarity"


def test_build_notification_engagement_flow(monkeypatch):
    monkeypatch.setattr(service_module, "get_user_response_rate", lambda user_id: {
        "user_id": user_id,
        "questions_sent": 10,
        "questions_answered": 6,
        "response_percentage": 60.0,
        "notification_type": "POSITIVE",
    })
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Nia",
        "language_code": "en",
        "video_language_id": 1,
    })

    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())
    result = service.build_notification(NotificationRequest(user_id=953, flow="engagement", should_send=False))

    assert result.flow == "engagement"
    assert result.notification_title == "Growth Check"


def test_get_performance_contract(monkeypatch):
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "performance": [{"kii_id": 119, "kii_name": "KII 119", "daily_target": 3.0, "seven_day_target": 21.0, "seven_day_actual": 12.0, "performance_percentage": 57.14}],
        "improvement_area": {"kii_id": 119, "kii_name": "KII 119", "performance_percentage": 57.14},
    })

    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())
    result = service.get_performance(953)

    assert result["user_id"] == 953
    assert result["seven_day_performance"][0]["kii_id"] == 119
    assert result["weakest_kii"]["kii_name"] == "KII 119"
