import json

import pytest
import requests

from app.llm import notification_generator as llm
from app.notifications.sender import NotificationSender
import scripts.run_daily as run_daily


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = "response"
        self._body = body or {"ok": True}

    def json(self):
        return self._body


class FakeOpenAIResponses:
    def __init__(self, failures, content):
        self.failures = failures
        self.content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary OpenAI failure")
        return type("Response", (), {"output_text": self.content})()


class FakeOpenAIClient:
    def __init__(self, failures=0, content=None):
        self.responses = FakeOpenAIResponses(
            failures,
            content or json.dumps({
                "title": "Title",
                "description": "Description",
                "action": "Watch now",
            }),
        )


def test_openai_retries_temporary_failures_with_expected_backoff(monkeypatch):
    client = FakeOpenAIClient(failures=3)
    sleeps = []
    monkeypatch.setattr(llm, "_is_temporary_openai_error", lambda error: True)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    result = llm.NotificationGenerator(client=client).generate("prompt")

    assert result["title"] == "Title"
    assert client.responses.calls == 4
    assert sleeps == [1, 2, 4]


def test_openai_stops_after_three_retries(monkeypatch):
    client = FakeOpenAIClient(failures=4)
    sleeps = []
    monkeypatch.setattr(llm, "_is_temporary_openai_error", lambda error: True)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="temporary OpenAI failure"):
        llm.NotificationGenerator(client=client).generate("prompt")

    assert client.responses.calls == 4
    assert sleeps == [1, 2, 4]


def test_openai_validation_error_is_not_retried(monkeypatch):
    client = FakeOpenAIClient(content=json.dumps({"title": "", "description": "Description"}))
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    with pytest.raises(ValueError, match="requires non-empty title"):
        llm.NotificationGenerator(client=client).generate("prompt")

    assert client.responses.calls == 1
    assert sleeps == []


def test_sender_retries_temporary_failures_with_expected_backoff(monkeypatch):
    responses = [requests.Timeout("timeout"), requests.ConnectionError("connection"), FakeResponse()]
    sleeps = []

    def post(*args, **kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("app.notifications.sender.requests.post", post)
    monkeypatch.setattr("app.notifications.sender.time.sleep", sleeps.append)

    result = NotificationSender(remote_url="https://example.test").send(
        user_id=1,
        notification_type="VIDEO_RECOMMENDATION",
        title="Title",
        description="Description",
        reference_id=10,
    )

    assert result["status_code"] == 200
    assert sleeps == [1, 2]


def test_sender_does_not_retry_permanent_http_failure(monkeypatch):
    calls = []
    sleeps = []

    def post(*args, **kwargs):
        calls.append(1)
        return FakeResponse(status_code=400)

    monkeypatch.setattr("app.notifications.sender.requests.post", post)
    monkeypatch.setattr("app.notifications.sender.time.sleep", sleeps.append)

    with pytest.raises(ValueError, match="Remote API error 400"):
        NotificationSender(remote_url="https://example.test").send(
            user_id=1,
            notification_type="VIDEO_RECOMMENDATION",
            title="Title",
            description="Description",
            reference_id=10,
        )

    assert len(calls) == 1
    assert sleeps == []


class FakeScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class FakeUserResult:
    def __init__(self, user_ids):
        self.user_ids = user_ids

    def __iter__(self):
        return iter([(user_id,) for user_id in self.user_ids])


class FakeSchedulerConnection:
    def __init__(self, user_ids, lock_acquired=True):
        self.user_ids = user_ids
        self.lock_acquired = lock_acquired
        self.queries = []

    def execute(self, query, params=None):
        query_text = str(query)
        self.queries.append(query_text)
        if "pg_try_advisory_lock" in query_text:
            return FakeScalarResult(self.lock_acquired)
        if "pg_advisory_unlock" in query_text:
            return FakeScalarResult(True)
        return FakeUserResult(self.user_ids)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeSchedulerEngine:
    def __init__(self, user_ids, lock_acquired=True):
        self.connection = FakeSchedulerConnection(user_ids, lock_acquired)

    def connect(self):
        return self.connection


def test_scheduler_rejects_second_process_and_releases_lock(monkeypatch):
    engine = FakeSchedulerEngine([1], lock_acquired=False)
    processed = []
    monkeypatch.setattr(run_daily, "engine", engine)
    monkeypatch.setattr(run_daily, "NotificationService", lambda: processed)
    monkeypatch.setattr(run_daily, "get_next_notification_for_user", lambda *args, **kwargs: processed.append("unexpected"))

    run_daily.main()

    assert processed == []
    assert any("pg_try_advisory_lock" in query for query in engine.connection.queries)
    assert not any("pg_advisory_unlock" in query for query in engine.connection.queries)


def test_scheduler_releases_lock_after_processing(monkeypatch):
    engine = FakeSchedulerEngine([])
    monkeypatch.setattr(run_daily, "engine", engine)
    monkeypatch.setattr(run_daily, "NotificationService", lambda: object())

    run_daily.main()

    assert any("pg_try_advisory_lock" in query for query in engine.connection.queries)
    assert any("pg_advisory_unlock" in query for query in engine.connection.queries)


def test_scheduler_continues_after_one_user_failure(monkeypatch):
    engine = FakeSchedulerEngine([1, 2, 3])
    processed = []

    class FakeResult:
        remote_send_status = "sent"
        notification_title = "Title"
        notification_body = "Body"
        reference_id = 10

        def model_dump(self, exclude_none=True):
            return {"notification_type": "VIDEO_RECOMMENDATION"}

    class FakeService:
        def build_notification(self, request):
            processed.append(request.user_id)
            if request.user_id == 2:
                raise RuntimeError("user 2 failed")
            return FakeResult()

    monkeypatch.setattr(run_daily, "engine", engine)
    monkeypatch.setattr(run_daily, "NotificationService", lambda: FakeService())
    monkeypatch.setattr(run_daily, "get_next_notification_for_user", lambda *args, **kwargs: "VIDEO_RECOMMENDATION")
    monkeypatch.setattr(run_daily, "has_notification_for_user_on_date", lambda *args, **kwargs: False)
    monkeypatch.setattr(run_daily, "insert_notification", lambda **kwargs: None)

    run_daily.main()

    assert processed == [1, 2, 3]
    assert any("pg_advisory_unlock" in query for query in engine.connection.queries)
