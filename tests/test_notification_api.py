from fastapi.testclient import TestClient

from app.main import app, service


client = TestClient(app)


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


def test_sentiment_post_processes_notification(monkeypatch):
    sentiment_data = {
        "user_id": 953,
        "notification": {
            "response_id": 10,
            "question_id": 1,
            "answer_id": 3,
            "question": "How is work going?",
            "answer": "Well",
        },
    }

    monkeypatch.setattr(
        service,
        "process_sentiment_notification",
        lambda user_id: sentiment_data,
    )
    monkeypatch.setattr(
        service,
        "build_notification",
        lambda _: (_ for _ in ()).throw(AssertionError("wrong notification path")),
    )

    response = client.post(
        "/notification/send",
        json={"user_id": 953, "flow": "sentiment"},
    )

    assert response.status_code == 200
    assert response.json() == sentiment_data


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