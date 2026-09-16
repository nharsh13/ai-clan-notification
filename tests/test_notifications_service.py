import app.notifications.service as service_module
from app.notifications.models import NotificationRequest


class DummySender:
    def __init__(self):
        self.remote_url = "https://example.test/notify"
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "ok", "payload": kwargs}


class DummyGenerator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return {
            "title": "Growth Check",
            "description": "Keep working on the area that needs the most attention.",
            "action": "Watch now",
        }


def test_build_notification_performance_flow(monkeypatch):
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "performance": [{"kii_id": 117, "kii_name": "KII 117", "daily_target": 2.0, "seven_day_target": 14.0, "seven_day_actual": 10.0, "performance_percentage": 71.43}],
        "improvement_area": {"kii_id": 117, "kii_name": "KII 117", "performance_percentage": 71.43},
    })
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "app_language_code": "hi",
        "video_language_ids": [1],
    })
    captured = {}

    def fake_recommend_video(performance, language_id, embed, db_engine, user_id):
        captured["kii_name"] = performance.kii_name
        captured["language_id"] = language_id
        captured["embedding"] = embed
        return {"video_id": 55, "title": "Focus on clarity", "creator_name": "Coach"}

    monkeypatch.setattr(service_module, "recommend_video", fake_recommend_video)

    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())
    result = service.build_notification(NotificationRequest(user_id=953, flow="performance", should_send=False))

    assert result.user_id == 953
    assert result.flow == "performance"
    assert result.notification_title == "Growth Check"
    assert result.video_id == 55
    assert result.video_title == "Focus on clarity"
    assert result.reference_id == 55
    assert result.deep_link == "/videos/55"
    assert result.video_popup is True
    assert captured == {"kii_name": "KII 117", "language_id": [1], "embedding": service_module.embed_text}


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
        "app_language_code": "en",
        "video_language_ids": [1],
    })

    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())
    result = service.build_notification(NotificationRequest(user_id=953, flow="engagement", should_send=False))

    assert result.flow == "engagement"
    assert result.notification_type == "SENTIMENT_ENGAGEMENT"
    assert result.notification_title == "Growth Check"


def test_build_notification_uses_exact_uppercase_scheduler_event_types(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "app_language_code": "en",
        "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 10},
    })
    monkeypatch.setattr(service_module, "recommend_video", lambda *args, **kwargs: {
        "video_id": 55, "title": "Focus better",
    })
    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())

    video_result = service.build_notification(NotificationRequest(user_id=953, flow="performance", should_send=False))
    assert video_result.notification_type == "VIDEO_RECOMMENDATION"

    monkeypatch.setattr(service_module, "get_user_response_rate", lambda user_id: {
        "user_id": user_id,
        "questions_sent": 10,
        "questions_answered": 6,
        "response_percentage": 60.0,
        "notification_type": "POSITIVE",
    })
    engagement_result = service.build_notification(NotificationRequest(user_id=953, flow="engagement", should_send=False))
    assert engagement_result.notification_type == "SENTIMENT_ENGAGEMENT"

    import app.sentiment.sentiment as sentiment_module

    monkeypatch.setattr(sentiment_module, "prepare_user_qa", lambda user_id: {"user_id": user_id, "questions": [{"question_id": 1, "responses": [{"question": "How is work going?", "answer": "Well"}]}]})
    sentiment_result = service.build_notification(NotificationRequest(user_id=953, flow="sentiment", should_send=False))
    assert sentiment_result.notification_type == "SENTIMENT_QA"


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


def test_performance_flow_passes_performance_query_embedding(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "app_language_code": "fr",
        "video_language_ids": [4, 7],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {
            "kii_id": 121,
            "kii_name": "Communication",
            "performance_percentage": 38.1,
        }
    })
    captured = {}

    def fake_embed(text):
        captured["query"] = text
        return [0.25] * 384

    monkeypatch.setattr(service_module, "embed_text", fake_embed)

    def fake_recommend(performance, language_id, embed, db_engine, user_id):
        captured["kii"] = performance.kii_id
        captured["language"] = language_id
        captured["embedding"] = embed("query probe")
        return {"video_id": 88, "title": "Communicate clearly"}

    monkeypatch.setattr(service_module, "recommend_video", fake_recommend)
    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())

    result = service.build_notification(NotificationRequest(user_id=953, flow="performance", should_send=False))

    assert result.video_id == 88
    assert result.deep_link == "/videos/88"
    assert captured == {
        "query": "query probe",
        "kii": 121,
        "language": [4, 7],
        "embedding": [0.25] * 384,
    }


def test_performance_flow_handles_no_video(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 0},
    })
    monkeypatch.setattr(service_module, "recommend_video", lambda *args, **kwargs: None)
    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())

    import pytest
    with pytest.raises(ValueError, match="No suitable video"):
        service.build_notification(NotificationRequest(user_id=953, flow="performance", should_send=False))


def test_performance_flow_propagates_llm_failure(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 10},
    })
    monkeypatch.setattr(service_module, "recommend_video", lambda *args, **kwargs: {
        "video_id": 55, "title": "Focus better",
    })

    class FailingGenerator:
        def generate(self, prompt):
            raise RuntimeError("LLM unavailable")

    service = service_module.NotificationService(sender=DummySender(), generator=FailingGenerator())
    import pytest
    with pytest.raises(RuntimeError, match="LLM unavailable"):
        service.build_notification(NotificationRequest(user_id=953, flow="performance", should_send=False))


def test_sender_receives_selected_video_reference(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 10},
    })
    monkeypatch.setattr(service_module, "recommend_video", lambda *args, **kwargs: {
        "video_id": 363, "title": "Focus better",
    })
    sender = DummySender()
    service = service_module.NotificationService(sender=sender, generator=DummyGenerator())

    result = service.build_notification(
        NotificationRequest(user_id=953, flow="performance", should_send=True)
    )

    assert result.remote_send_status == "sent"
    assert result.reference_id == 363
    assert result.deep_link == "/videos/363"
    assert sender.calls[0]["reference_id"] == 363
    assert sender.calls[0]["video_popup"] is True


def test_app_language_reaches_llm_but_video_languages_reach_recommender(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "app_language_code": "ta",
        "video_language_ids": [2, 5],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {
            "kii_id": 117,
            "kii_name": "Focus",
            "performance_percentage": 10,
        },
    })
    recommender_languages = []

    def fake_recommend(performance, language_id, embed, db_engine, user_id):
        recommender_languages.append(language_id)
        return {"video_id": 701, "title": "Focus better"}

    monkeypatch.setattr(service_module, "recommend_video", fake_recommend)
    generator = DummyGenerator()
    service = service_module.NotificationService(sender=DummySender(), generator=generator)

    result = service.build_notification(
        NotificationRequest(user_id=953, flow="performance", should_send=False)
    )

    assert result.reference_id == 701
    assert result.deep_link == "/videos/701"
    assert recommender_languages == [[2, 5]]
    assert "Notification language: ta" in generator.prompts[0]


def test_performance_flow_normalizes_numeric_creator_identifier(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 10},
    })
    monkeypatch.setattr(service_module, "recommend_video", lambda *args, **kwargs: {
        "video_id": 55, "title": "Focus better", "creator_name": 1877,
    })

    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())
    result = service.build_notification(
        NotificationRequest(user_id=953, flow="performance", should_send=False)
    )

    assert result.creator_name == "1877"


def _eligible_sentiment_rows():
    return [
        {
            "response_id": 11,
            "user_id": 953,
            "question_id": 1,
            "question": "How do you respond to feedback?",
            "answer_id": 3,
            "answer": "I listen and improve.",
        },
        {
            "response_id": 12,
            "user_id": 953,
            "question_id": 2,
            "question": "How do you handle change?",
            "answer_id": 5,
            "answer": "I adapt and communicate.",
        },
    ]


def _configure_sentiment_service(monkeypatch, sender, generator):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "app_language_code": "en",
        "video_language_ids": [1],
    })
    rows = _eligible_sentiment_rows()
    monkeypatch.setattr(service_module, "get_eligible_user_qa", lambda user_id, db_engine=None: rows)
    monkeypatch.setattr(
        service_module,
        "prepare_user_qa",
        lambda user_id, db_engine=None, rows=None: {
            "user_id": user_id,
            "questions": [
                {
                    "question_id": row["question_id"],
                    "responses": [{"question": row["question"], "answer": row["answer"]}],
                }
                for row in rows
            ],
        },
    )
    return service_module.NotificationService(sender=sender, generator=generator)


def test_sentiment_saves_all_eligible_history_after_successful_send(monkeypatch):
    saved = []
    monkeypatch.setattr(
        service_module,
        "save_sentiment_notification_history",
        lambda records, db_engine: saved.extend(records),
    )
    service = _configure_sentiment_service(monkeypatch, DummySender(), DummyGenerator())

    result = service.build_notification(NotificationRequest(user_id=953, flow="sentiment"))

    assert result.notification_type == "SENTIMENT_QA"
    assert [row["response_id"] for row in saved] == [11, 12]
    assert [row["question_id"] for row in saved] == [1, 2]
    assert [row["answer_id"] for row in saved] == [3, 5]


def test_sentiment_with_no_eligible_qa_skips_llm_and_sender(monkeypatch):
    generator = DummyGenerator()
    sender = DummySender()
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "get_eligible_user_qa", lambda user_id, db_engine=None: [])

    result = service_module.NotificationService(sender=sender, generator=generator).build_notification(
        NotificationRequest(user_id=953, flow="sentiment")
    )

    assert result is None
    assert generator.prompts == []
    assert sender.calls == []


def test_sentiment_llm_failure_does_not_save_history(monkeypatch):
    saved = []
    monkeypatch.setattr(service_module, "save_sentiment_notification_history", saved.extend)

    class FailingGenerator:
        def generate(self, prompt):
            raise RuntimeError("LLM unavailable")

    service = _configure_sentiment_service(monkeypatch, DummySender(), FailingGenerator())

    import pytest
    with pytest.raises(RuntimeError, match="LLM unavailable"):
        service.build_notification(NotificationRequest(user_id=953, flow="sentiment"))
    assert saved == []


def test_sentiment_send_failure_does_not_save_history(monkeypatch):
    saved = []
    monkeypatch.setattr(service_module, "save_sentiment_notification_history", saved.extend)

    class FailingSender(DummySender):
        def send(self, **kwargs):
            raise RuntimeError("sender unavailable")

    service = _configure_sentiment_service(monkeypatch, FailingSender(), DummyGenerator())
    result = service.build_notification(NotificationRequest(user_id=953, flow="sentiment"))

    assert result.remote_send_status == "failed"
    assert saved == []


def test_sentiment_history_failure_is_reported_as_failed(monkeypatch):
    def fail_history(*args, **kwargs):
        raise RuntimeError("history insert failed")

    monkeypatch.setattr(service_module, "save_sentiment_notification_history", fail_history)
    service = _configure_sentiment_service(monkeypatch, DummySender(), DummyGenerator())

    result = service.build_notification(NotificationRequest(user_id=953, flow="sentiment"))

    assert result.remote_send_status == "failed"
    assert result.error == "history insert failed"
