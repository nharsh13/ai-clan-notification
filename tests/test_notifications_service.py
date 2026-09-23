import asyncio

import pytest

import app.notifications.service as service_module
import scripts.run_daily as run_daily
from app.notifications.models import NotificationRequest


class DummySender:
    def __init__(self):
        self.remote_url = "https://example.test/notify"
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "ok", "payload": kwargs}

    async def send_async(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "ok", "payload": kwargs}


class DummyGenerator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return {
            "title": "Growth Check",
            "description": "Keep working on the area that needs the most attention.",
            "action": "Watch now",
        }

    async def generate_async(self, prompt, **kwargs):
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
    assert result.title == "Growth Check"
    assert result.reference_id == 55
    assert result.reference_id == 55
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
    assert result.title == "Growth Check"


def test_build_notification_engagement_skips_zero_questions(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Nia",
        "app_language_code": "en",
        "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "get_user_response_rate", lambda user_id: None)
    sender = DummySender()
    generator = DummyGenerator()

    service = service_module.NotificationService(
        sender=sender,
        generator=generator,
    )
    result = service.build_notification(
        NotificationRequest(user_id=953, flow="engagement")
    )

    assert result is not None
    assert result.reason == "NO_ENGAGEMENT_DATA"
    assert "Keep Building Momentum" in result.title
    assert generator.prompts == []
    assert len(sender.calls) == 1
    assert sender.calls[0]["notification_type"] == "SENTIMENT_ENGAGEMENT"


def test_build_notification_engagement_keeps_positive_threshold(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Nia",
        "app_language_code": "en",
        "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "get_user_response_rate", lambda user_id: {
        "user_id": user_id,
        "questions_sent": 10,
        "questions_answered": 6,
        "response_percentage": 60.0,
        "notification_type": "POSITIVE",
    })

    result = service_module.NotificationService(
        sender=DummySender(),
        generator=DummyGenerator(),
    ).build_notification(
        NotificationRequest(user_id=953, flow="engagement", should_send=False)
    )

    assert result is not None
    assert result.notification_type == "SENTIMENT_ENGAGEMENT"


def test_engagement_does_not_require_video_language_or_recommendation(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Nia",
        "app_language_code": "en",
        "video_language_ids": [],
    })
    monkeypatch.setattr(service_module, "get_user_response_rate", lambda user_id: {
        "user_id": user_id,
        "questions_sent": 10,
        "questions_answered": 6,
        "response_percentage": 60.0,
        "notification_type": "POSITIVE",
    })
    monkeypatch.setattr(
        service_module,
        "recommend_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("engagement must not recommend a video")
        ),
    )

    result = service_module.NotificationService(
        sender=DummySender(),
        generator=DummyGenerator(),
    ).build_notification(NotificationRequest(user_id=953, flow="engagement"))

    assert result.notification_type == "SENTIMENT_ENGAGEMENT"
    assert result.reference_id == 0
    assert result.video_popup is None


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

    assert result.reference_id == 88
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

    result = service.build_notification(
        NotificationRequest(user_id=953, flow="performance", should_send=False)
    )

    assert result is not None
    assert result.reason == "NO_VIDEO_RECOMMENDATION"
    assert result.title.startswith("Ava")
    assert service.generator.prompts == []
    assert service.sender.calls == []


def test_performance_flow_can_fallback_when_no_video_and_reason_is_tracked(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 0},
    })
    monkeypatch.setattr(service_module, "recommend_video", lambda *args, **kwargs: None)
    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())

    result = service.build_notification(
        NotificationRequest(user_id=953, flow="performance", should_send=False),
        allow_fallback=True,
    )

    assert result is not None
    assert result.notification_type == "VIDEO_RECOMMENDATION"
    assert result.reason == "NO_VIDEO_RECOMMENDATION"
    assert result.title.startswith("Ava")
    assert result.description


def test_sentiment_flow_calls_llm_when_no_qa_and_sends_generated_notification(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "get_eligible_user_qa", lambda user_id, db_engine=None: [])
    sender = DummySender()
    generator = DummyGenerator()
    service = service_module.NotificationService(sender=sender, generator=generator)

    result = service.build_notification(
        NotificationRequest(user_id=953, flow="sentiment", should_send=True),
        allow_fallback=True,
    )

    assert result is not None
    assert result.notification_type == "SENTIMENT_QA"
    assert result.reference_id == 0
    assert result.video_popup is None
    assert result.image is None
    assert result.reason is None
    assert len(generator.prompts) == 1
    assert "daily questions" in generator.prompts[0].lower()
    assert len(sender.calls) == 1
    assert sender.calls[0]["notification_type"] == "SENTIMENT_QA"


def test_async_performance_fallback_is_active_by_default(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 0},
    })
    monkeypatch.setattr(service_module, "recommend_video", lambda *args, **kwargs: None)

    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())
    result = asyncio.run(service.build_notification_async(NotificationRequest(user_id=953, flow="performance", should_send=False)))

    assert result is not None
    assert result.notification_type == "VIDEO_RECOMMENDATION"
    assert result.reason == "NO_VIDEO_RECOMMENDATION"
    assert result.title.startswith("Ava")


def test_async_sentiment_no_qa_calls_llm_and_sends_generated_notification(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "get_eligible_user_qa", lambda user_id, db_engine=None: [])

    sender = DummySender()
    generator = DummyGenerator()
    service = service_module.NotificationService(sender=sender, generator=generator)
    result = asyncio.run(service.build_notification_async(NotificationRequest(user_id=953, flow="sentiment", should_send=True)))

    assert result is not None
    assert result.notification_type == "SENTIMENT_QA"
    assert result.reference_id == 0
    assert result.video_popup is None
    assert result.image is None
    assert result.reason is None
    assert len(generator.prompts) == 1
    assert "daily questions" in generator.prompts[0].lower()
    assert len(sender.calls) == 1
    assert sender.calls[0]["notification_type"] == "SENTIMENT_QA"


def test_async_engagement_no_data_falls_back_without_special_flag(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "get_user_response_rate", lambda user_id: None)

    service = service_module.NotificationService(sender=DummySender(), generator=DummyGenerator())
    result = asyncio.run(service.build_notification_async(NotificationRequest(user_id=953, flow="engagement", should_send=False)))

    assert result is not None
    assert result.notification_type == "SENTIMENT_ENGAGEMENT"
    assert result.reason == "NO_ENGAGEMENT_DATA"
    assert "Keep Building Momentum" in result.title


@pytest.mark.parametrize("video_language_id", [9, 3])
def test_performance_flow_keeps_valid_hindi_and_telugu_recommendations(
    monkeypatch,
    video_language_id,
):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "app_language_code": "en",
        "video_language_ids": [video_language_id],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 0},
    })
    monkeypatch.setattr(service_module, "recommend_video", lambda *args, **kwargs: {
        "video_id": 55,
        "title": "Focus better",
    })

    result = service_module.NotificationService(
        sender=DummySender(),
        generator=DummyGenerator(),
    ).build_notification(
        NotificationRequest(user_id=953, flow="performance", should_send=False)
    )

    assert result is not None
    assert result.reference_id == 55
    assert result.notification_type == "VIDEO_RECOMMENDATION"


@pytest.mark.parametrize("video_language_id", [9, 3])
def test_performance_flow_skips_when_hindi_or_telugu_video_is_unavailable(
    monkeypatch,
    video_language_id,
):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "app_language_code": "en",
        "video_language_ids": [video_language_id],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 0},
    })
    monkeypatch.setattr(service_module, "recommend_video", lambda *args, **kwargs: None)
    sender = DummySender()
    generator = DummyGenerator()

    service = service_module.NotificationService(
        sender=sender,
        generator=generator,
    )
    result = service.build_notification(
        NotificationRequest(user_id=953, flow="performance", should_send=False)
    )

    assert result is not None
    assert result.reason == "NO_VIDEO_RECOMMENDATION"
    assert result.title.startswith("Ava")
    assert generator.prompts == []
    assert sender.calls == []


def test_performance_flow_defaults_null_video_language_to_telugu(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "app_language_code": "en",
        "video_language_ids": None,
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 0},
    })
    captured = {}
    monkeypatch.setattr(
        service_module.NotificationService,
        "_get_telugu_video_language_id",
        lambda self: 3,
    )
    monkeypatch.setattr(
        service_module,
        "recommend_video",
        lambda performance, language_id, embed, db_engine, user_id: (
            captured.update(language_id=language_id) or {"video_id": 55, "title": "Focus better"}
        ),
    )
    sender = DummySender()
    generator = DummyGenerator()

    service = service_module.NotificationService(
        sender=sender,
        generator=generator,
    )
    result = service.build_notification(
        NotificationRequest(user_id=953, flow="performance", should_send=False)
    )

    assert result is not None
    assert captured["language_id"] == [3]
    assert result.reference_id == 55
    assert sender.calls == []


def test_performance_flow_defaults_empty_video_language_to_telugu(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "app_language_code": "en",
        "video_language_ids": [],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 0},
    })
    captured = {}
    monkeypatch.setattr(
        service_module.NotificationService,
        "_get_telugu_video_language_id",
        lambda self: 3,
    )
    monkeypatch.setattr(
        service_module,
        "recommend_video",
        lambda performance, language_id, embed, db_engine, user_id: (
            captured.update(language_id=language_id) or {"video_id": 55, "title": "Focus better"}
        ),
    )

    result = service_module.NotificationService(
        sender=DummySender(),
        generator=DummyGenerator(),
    ).build_notification(
        NotificationRequest(user_id=953, flow="performance")
    )

    assert result is not None
    assert captured["language_id"] == [3]
    assert result.reference_id == 55


def test_telugu_video_language_id_is_loaded_from_language_table():
    class FakeResult:
        def scalar_one_or_none(self):
            return 3

    class FakeConnection:
        def execute(self, query):
            assert "public.md_language" in str(query)
            assert "lower(code) = 'te'" in str(query)
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeEngine:
        def connect(self):
            return FakeConnection()

    service = service_module.NotificationService(db_engine=FakeEngine())

    assert service._get_telugu_video_language_id() == 3


@pytest.mark.parametrize("recommendation", [
    {"video_id": 88, "title": "English focus"},
    None,
])
def test_performance_flow_supports_configured_future_video_language(
    monkeypatch,
    recommendation,
):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "app_language_code": "en",
        "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 0},
    })
    monkeypatch.setattr(
        service_module,
        "recommend_video",
        lambda performance, language_id, embed, db_engine, user_id: recommendation,
    )
    generator = DummyGenerator()
    sender = DummySender()

    result = service_module.NotificationService(
        sender=sender,
        generator=generator,
    ).build_notification(
        NotificationRequest(user_id=953, flow="performance", should_send=False)
    )

    if recommendation is None:
        assert result is not None
        assert result.reason == "NO_VIDEO_RECOMMENDATION"
        assert result.title.startswith("Ava")
        assert generator.prompts == []
    else:
        assert result is not None
        assert result.reference_id == 88
        assert result.notification_type == "VIDEO_RECOMMENDATION"


def test_performance_flow_does_not_send_when_query_embedding_fails(monkeypatch):
    import pytest

    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 0},
    })

    def failing_recommendation(*args, **kwargs):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(service_module, "recommend_video", failing_recommendation)
    sender = DummySender()
    service = service_module.NotificationService(sender=sender, generator=DummyGenerator())

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        service.build_notification(NotificationRequest(user_id=953, flow="performance"))

    assert sender.calls == []


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
        def generate(self, prompt, **kwargs):
            raise RuntimeError("LLM unavailable")

    sender = DummySender()
    service = service_module.NotificationService(sender=sender, generator=FailingGenerator())
    import pytest
    result = service.build_notification(NotificationRequest(user_id=953, flow="performance"))
    assert result is None
    assert service.last_skip_reason == "LLM_NO_RESPONSE"
    assert sender.calls == []


def test_empty_llm_response_is_skipped_with_reason(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {
        "improvement_area": {"kii_id": 117, "kii_name": "Focus", "performance_percentage": 10},
    })
    monkeypatch.setattr(service_module, "recommend_video", lambda *args, **kwargs: {
        "video_id": 55, "title": "Focus better",
    })

    class EmptyGenerator:
        def generate(self, prompt, **kwargs):
            return {}

    service = service_module.NotificationService(sender=DummySender(), generator=EmptyGenerator())
    result = service.build_notification(NotificationRequest(user_id=953, flow="performance"))

    assert result is None
    assert service.last_skip_reason == "LLM_NO_RESPONSE"
    assert service.sender.calls == []


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
    assert sender.calls[0]["reference_id"] == 363
    assert sender.calls[0]["video_popup"] is True
    assert set(sender.calls[0]) == {
        "user_id",
        "notification_type",
        "title",
        "description",
        "reference_id",
        "video_popup",
        "image",
    }


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
    assert recommender_languages == [[2, 5]]
    assert "Language: ta" in generator.prompts[0]


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


def test_sentiment_with_no_eligible_qa_calls_llm_and_raises_on_failure(monkeypatch):
    class FailingGenerator:
        def generate(self, prompt, **kwargs):
            raise ValueError("LLM unavailable")

    sender = DummySender()
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava", "app_language_code": "en", "video_language_ids": [1],
    })
    monkeypatch.setattr(service_module, "get_eligible_user_qa", lambda user_id, db_engine=None: [])

    service = service_module.NotificationService(sender=sender, generator=FailingGenerator())
    result = service.build_notification(
        NotificationRequest(user_id=953, flow="sentiment")
    )

    assert result is None
    assert service.last_skip_reason == "LLM_NO_RESPONSE"
    assert sender.calls == []


def test_sentiment_does_not_require_video_language_or_recommendation(monkeypatch):
    monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: {
        "user_name": "Ava",
        "app_language_code": "en",
        "video_language_ids": [],
    })
    monkeypatch.setattr(service_module, "get_eligible_user_qa", lambda user_id, db_engine=None: _eligible_sentiment_rows())
    monkeypatch.setattr(
        service_module,
        "prepare_user_qa",
        lambda user_id, db_engine=None, rows=None: {
            "user_id": user_id,
            "questions": [{
                "question_id": row["question_id"],
                "responses": [{"question": row["question"], "answer": row["answer"]}],
            } for row in rows],
        },
    )
    monkeypatch.setattr(
        service_module,
        "recommend_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Q&A must not recommend a video")
        ),
    )

    result = service_module.NotificationService(
        sender=DummySender(),
        generator=DummyGenerator(),
    ).build_notification(NotificationRequest(user_id=953, flow="sentiment"))

    assert result.notification_type == "SENTIMENT_QA"
    assert result.reference_id == 0
    assert result.video_popup is None


def test_get_sentiment_returns_unused_qa(monkeypatch):
    monkeypatch.setattr(service_module, "get_eligible_user_qa", lambda user_id, db_engine=None: [{
        "user_id": 953,
        "question_id": 4,
        "question": "How do you handle feedback?",
        "answer_id": 8,
        "answer": "I listen carefully.",
        "selection_type": "NEW",
    }])

    service = service_module.NotificationService()

    assert service.get_sentiment(953) == {
        "user_id": 953,
        "responses": [{
            "question_id": 4,
            "question": "How do you handle feedback?",
            "answer_id": 8,
            "answer": "I listen carefully.",
        }],
    }


def test_get_sentiment_reuses_qa_when_all_are_exhausted(monkeypatch):
    monkeypatch.setattr(service_module, "get_eligible_user_qa", lambda user_id, db_engine=None: [{
        "user_id": 953,
        "question_id": 3,
        "question": "What kind of customers do you like most?",
        "answer_id": 9,
        "answer": "Curious customers.",
        "selection_type": "REUSED",
    }])

    service = service_module.NotificationService()

    assert service.get_sentiment(953)["responses"] == [{
        "question_id": 3,
        "question": "What kind of customers do you like most?",
        "answer_id": 9,
        "answer": "Curious customers.",
    }]


def test_sentiment_llm_failure_does_not_save_history(monkeypatch):
    saved = []
    monkeypatch.setattr(service_module, "save_sentiment_notification_history", saved.extend)

    class FailingGenerator:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("LLM unavailable")

    service = _configure_sentiment_service(monkeypatch, DummySender(), FailingGenerator())

    import pytest
    result = service.build_notification(NotificationRequest(user_id=953, flow="sentiment"))
    assert result is None
    assert service.last_skip_reason == "LLM_NO_RESPONSE"
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


def test_concurrent_performance_notifications_are_isolated_by_user(monkeypatch):
    async def run_test():
        user_profiles = {
            101: {"user_name": "USER_A", "app_language_code": "en", "video_language_ids": [11], "weakest_kii": {"kii_name": "Focus", "kii_id": 1001}},
            102: {"user_name": "USER_B", "app_language_code": "hi", "video_language_ids": [22], "weakest_kii": {"kii_name": "Consistency", "kii_id": 1002}},
            103: {"user_name": "USER_C", "app_language_code": "ta", "video_language_ids": [33], "weakest_kii": {"kii_name": "Planning", "kii_id": 1003}},
            104: {"user_name": "USER_D", "app_language_code": "te", "video_language_ids": [44], "weakest_kii": {"kii_name": "Communication", "kii_id": 1004}},
            105: {"user_name": "USER_E", "app_language_code": "fr", "video_language_ids": [55], "weakest_kii": {"kii_name": "Execution", "kii_id": 1005}},
        }

        monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: user_profiles[user_id])
        monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {"performance": [], "improvement_area": user_profiles[user_id]["weakest_kii"]})
        monkeypatch.setattr(
            service_module,
            "recommend_video",
            lambda performance, language_id, embed, db_engine, user_id: {"video_id": user_id * 10, "title": f"Video for {user_profiles[user_id]['user_name']}"},
        )

        prompt_records = {}
        payload_records = {}

        class RecordingGenerator:
            def generate(self, prompt, **kwargs):
                user_id = kwargs.get("user_id")
                prompt_records[user_id] = prompt
                parsed_name = prompt.split("User: ", 1)[1].splitlines()[0].strip()
                return {
                    "title": f"Hello {parsed_name}, keep moving",
                    "description": f"Your focus area is {user_profiles[user_id]['weakest_kii']['kii_name']}",
                    "action": "Watch now",
                }

            async def generate_async(self, prompt, **kwargs):
                return self.generate(prompt, **kwargs)

        class RecordingSender:
            def __init__(self):
                self.calls = []
                self.remote_url = "https://example.test/notify"

            def send(self, **kwargs):
                payload_records[kwargs["user_id"]] = kwargs
                self.calls.append(kwargs)
                return {"status": "ok", "payload": kwargs}

            async def send_async(self, **kwargs):
                return self.send(**kwargs)

        async def process_user(user_id: int):
            sender = RecordingSender()
            generator = RecordingGenerator()
            service = service_module.NotificationService(sender=sender, generator=generator)
            result = await service.build_notification_async(NotificationRequest(user_id=user_id, flow="performance", should_send=True))
            return {
                "user_id": result.user_id,
                "user_name": user_profiles[user_id]["user_name"],
                "prompt": prompt_records[user_id],
                "payload": payload_records[user_id],
                "title": result.title,
                "description": result.description,
            }

        results = await asyncio.gather(*(process_user(user_id) for user_id in sorted(user_profiles)))

        for user_id in sorted(user_profiles):
            record = next(item for item in results if item["user_id"] == user_id)
            profile = user_profiles[user_id]
            prompt = record["prompt"]
            payload = record["payload"]

            assert profile["user_name"] in prompt
            assert f"User: {profile['user_name']}" in prompt
            assert profile["app_language_code"].upper() in prompt or profile["app_language_code"].lower() in prompt
            assert profile["weakest_kii"]["kii_name"] in prompt or profile["weakest_kii"]["kii_name"] in record["description"]
            assert payload["user_id"] == user_id
            assert payload["notification_type"] == "VIDEO_RECOMMENDATION"
            assert profile["user_name"] in payload["title"]

            for other_user_id, other_profile in user_profiles.items():
                if other_user_id == user_id:
                    continue
                assert other_profile["user_name"] not in prompt
                assert other_profile["user_name"] not in payload["title"]
                assert other_profile["user_name"] not in payload["description"]

        assert run_daily.MAX_CONCURRENT_USERS == 5

    asyncio.run(run_test())


def test_scheduler_concurrency_preserves_each_user_context(monkeypatch):
    async def run_test():
        users = [(101, "USER_A"), (102, "USER_B"), (103, "USER_C"), (104, "USER_D"), (105, "USER_E")]
        seen = []

        monkeypatch.setattr(run_daily, "_collect_users", lambda: users)
        monkeypatch.setattr(run_daily, "get_next_notification_for_user", lambda user_id, db_engine=None, **kwargs: "VIDEO_RECOMMENDATION")
        monkeypatch.setattr(run_daily, "has_notification_for_user_on_date", lambda *args, **kwargs: False)

        async def fake_build_notification_async(self, request, *args, **kwargs):
            user_id = request.user_id
            seen.append({"user_id": user_id, "flow": request.flow})
            return type(
                "Result",
                (),
                    {
                        "user_id": user_id,
                    "notification_type": "VIDEO_RECOMMENDATION",
                    "remote_send_status": "sent",
                    "remote_send_response": {"status_code": 200},
                    "error": None,
                    "title": f"Hello {next(name for uid, name in users if uid == user_id)}, keep going",
                    "description": f"User {user_id} update",
                },
            )()

        monkeypatch.setattr(service_module.NotificationService, "build_notification_async", fake_build_notification_async)

        summary = await run_daily.run_scheduler_async(test_mode=True)

        assert summary["total_users"] == 5
        assert summary["successful_count"] == 5
        assert summary["failed_count"] == 0
        assert summary["skipped_count"] == 0
        assert {item["user_id"] for item in seen} == {101, 102, 103, 104, 105}
        assert len(seen) == 5

    asyncio.run(run_test())


def test_negative_cross_contamination_does_not_leak_other_user_names(monkeypatch):
    async def run_test():
        user_profiles = {
            101: {"user_name": "USER_A", "app_language_code": "en", "video_language_ids": [11], "weakest_kii": {"kii_name": "Focus", "kii_id": 1001}},
            102: {"user_name": "USER_B", "app_language_code": "hi", "video_language_ids": [22], "weakest_kii": {"kii_name": "Consistency", "kii_id": 1002}},
            103: {"user_name": "USER_C", "app_language_code": "ta", "video_language_ids": [33], "weakest_kii": {"kii_name": "Planning", "kii_id": 1003}},
        }

        monkeypatch.setattr(service_module, "get_user", lambda user_id, db_engine=None: user_profiles[user_id])
        monkeypatch.setattr(service_module, "calculate_performance", lambda user_id: {"performance": [], "improvement_area": user_profiles[user_id]["weakest_kii"]})
        monkeypatch.setattr(
            service_module,
            "recommend_video",
            lambda performance, language_id, embed, db_engine, user_id: {"video_id": user_id * 10, "title": f"Video for {user_profiles[user_id]['user_name']}"},
        )

        recorded = []

        class RecordingGenerator:
            def generate(self, prompt, **kwargs):
                name = prompt.split("User: ", 1)[1].splitlines()[0].strip()
                recorded.append({"user_id": kwargs["user_id"], "name": name, "prompt": prompt})
                return {"title": f"Hello {name}, keep moving", "description": f"Your focus is {user_profiles[kwargs['user_id']]['weakest_kii']['kii_name']}", "action": "Watch now"}

            async def generate_async(self, prompt, **kwargs):
                return self.generate(prompt, **kwargs)

        class RecordingSender:
            def __init__(self):
                self.calls = []
                self.remote_url = "https://example.test/notify"

            def send(self, **kwargs):
                self.calls.append(kwargs)
                return {"status": "ok", "payload": kwargs}

            async def send_async(self, **kwargs):
                return self.send(**kwargs)

        async def process_user(user_id: int):
            service = service_module.NotificationService(sender=RecordingSender(), generator=RecordingGenerator())
            await service.build_notification_async(NotificationRequest(user_id=user_id, flow="performance", should_send=True))

        await asyncio.gather(*(process_user(uid) for uid in sorted(user_profiles)))

        for entry in recorded:
            for other_name in {profile["user_name"] for profile in user_profiles.values()} - {entry["name"]}:
                assert other_name not in entry["prompt"]
                assert other_name not in entry["name"]

        assert {entry["name"] for entry in recorded} == {"USER_A", "USER_B", "USER_C"}

    asyncio.run(run_test())
