from fastapi.testclient import TestClient

from app.main import app, service
from app.notifications.models import NotificationResponse


client = TestClient(app)


def test_performance_send_matches_contract_and_forwards_selected_video(monkeypatch):
    def build_notification(request):
        assert request.user_id == 953
        assert request.campaign_day == 2
        return NotificationResponse(
            user_id=953,
            flow="performance",
            campaign_day=2,
            action="Watch now",
            audience_strategy="dynamic",
            cohort_key="ai_clan",
            creator_name="Coach",
            deep_link="/videos/55",
            notification_body="Keep building your communication skills.",
            notification_title="Ava, watch this next",
            notification_type="video_recommendation",
            should_send=True,
            video_id=55,
            video_title="Communicate clearly",
            reference_id=55,
            video_popup="Y",
            remote_send_status="sent",
            remote_send_response={
                "remote_url": "REMOTE_NOTIFICATION_SEND_URL",
                "request_payload": [{
                    "description": "Keep building your communication skills.",
                    "notification_type": "video_recommendation",
                    "reference_id": 55,
                    "title": "Ava, watch this next",
                    "user_id": 953,
                    "video_popup": "Y",
                }],
                "response": {},
                "status_code": 200,
            },
        )

    monkeypatch.setattr(service, "build_notification", build_notification)
    response = client.post("/notification/send", json={"campaign_day": 2, "user_id": 953})

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["user_id"] == 953
    assert set(result["notification"]) == {
        "action",
        "audience_strategy",
        "campaign_day",
        "cohort_key",
        "creator_name",
        "deep_link",
        "notification_body",
        "notification_title",
        "notification_type",
        "reference_id",
        "should_send",
        "video_id",
        "video_popup",
        "video_title",
    }
    assert result["notification"]["video_title"] == "Communicate clearly"
    assert result["remote_send_response"]["request_payload"][0]["user_id"] == 953


def test_performance_send_returns_gateway_error_on_remote_failure(monkeypatch):
    monkeypatch.setattr(
        service,
        "build_notification",
        lambda request: NotificationResponse(
            user_id=request.user_id,
            flow="performance",
            notification_title="Title",
            notification_body="Body",
            should_send=True,
            remote_send_status="failed",
            error="Remote API error 503",
        ),
    )

    response = client.post("/notification/send", json={"campaign_day": 2, "user_id": 953})

    assert response.status_code == 502
    assert response.json()["detail"] == "Remote API error 503"


def test_performance_endpoint_is_read_only(monkeypatch):
    def get_performance(user_id):
        assert user_id == 953
        return {
            "user_id": user_id,
            "seven_day_performance": [
                {
                    "kii_id": 117,
                    "kii_name": "KII 117",
                    "daily_target": 2.0,
                    "seven_day_target": 14.0,
                    "seven_day_actual": 10.0,
                    "performance_percentage": 71.43,
                }
            ],
            "weakest_kii": {
                "kii_id": 117,
                "kii_name": "KII 117",
                "performance_percentage": 71.43,
            },
        }

    monkeypatch.setattr(service, "get_performance", get_performance)
    monkeypatch.setattr(service, "build_notification", lambda _: (_ for _ in ()).throw(AssertionError()))

    response = client.get("/notification/performance", params={"user_id": 953})

    assert response.status_code == 200
    result = response.json()
    assert result["user_id"] == 953
    assert len(result["seven_day_performance"]) == 1
    assert result["seven_day_performance"][0]["seven_day_actual"] == 10.0
    assert result["weakest_kii"]["performance_percentage"] == 71.43


def test_engagement_endpoint_is_read_only(monkeypatch):
    def get_engagement(user_id):
        assert user_id == 953
        return {
            "user_id": user_id,
            "questions_sent": 10,
            "questions_answered": 6,
            "response_percentage": 60.0,
            "notification_type": "POSITIVE",
        }

    monkeypatch.setattr(service, "get_engagement", get_engagement)
    monkeypatch.setattr(service, "build_notification", lambda _: (_ for _ in ()).throw(AssertionError()))

    response = client.get("/notification/engagement", params={"user_id": 953})

    assert response.status_code == 200
    assert response.json()["response_percentage"] == 60.0
    assert response.json()["notification_type"] == "POSITIVE"


def test_sentiment_endpoint_is_read_only(monkeypatch):
    def get_sentiment(user_id):
        assert user_id == 953
        return {
            "user_id": user_id,
            "responses": [
                {
                    "question": "How is work going?",
                    "answer": "Well",
                    "what_it_conveys": "Positive",
                    "recommended_action_to_manager": "Recognize progress",
                }
            ],
        }

    monkeypatch.setattr(service, "get_sentiment", get_sentiment)
    monkeypatch.setattr(service, "build_notification", lambda _: (_ for _ in ()).throw(AssertionError()))

    response = client.get("/notification/sentiment", params={"user_id": 953})

    assert response.status_code == 200
    assert response.json()["responses"][0]["recommended_action_to_manager"] == "Recognize progress"


def test_send_rejects_internal_notification_fields():
    response = client.post(
        "/notification/send",
        json={"campaign_day": 2, "user_id": 953, "flow": "sentiment"},
    )

    assert response.status_code == 422


def test_sentiment_response_shape_is_qa_only(monkeypatch):
    class FakeResult:
        def mappings(self):
            return [
                {
                    "question_id": 1,
                    "question_text": "How is work going?",
                    "answer_id": 3,
                    "answer_text": "Well",
                },
                {
                    "question_id": 3,
                    "question_text": "What would help?",
                    "answer_id": 11,
                    "answer_text": "More feedback",
                },
            ]

    class FakeConnection:
        def execute(self, query, params):
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeEngine:
        def connect(self):
            return FakeConnection()

    from app.sentiment import sentiment

    responses = sentiment.get_responses(953, FakeEngine())

    assert responses == [
        {
            "question_id": 1,
            "question": "How is work going?",
            "answer_id": 3,
            "answer": "Well",
        },
        {
            "question_id": 3,
            "question": "What would help?",
            "answer_id": 11,
            "answer": "More feedback",
        },
    ]
    assert all(
        set(response) == {"question_id", "question", "answer_id", "answer"}
        for response in responses
    )