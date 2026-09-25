import asyncio
import json
import logging
import os

import pytest
import requests
import httpx
from openai import APITimeoutError

from app.llm import llm_client as llm
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
        self.kwargs = []

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
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


def test_scheduler_config_disables_hf_terminal_noise(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)

    run_daily.configure_terminal_logging()

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert logging.getLogger("huggingface_hub").level >= logging.ERROR
    assert logging.getLogger("sentence_transformers").level >= logging.ERROR
    assert logging.getLogger("transformers").level >= logging.ERROR


def test_openai_retries_temporary_failures_with_expected_backoff(monkeypatch):
    client = FakeOpenAIClient(failures=3)
    sleeps = []
    monkeypatch.setattr(llm, "_is_temporary_openai_error", lambda error: True)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    result = llm.NotificationGenerator(client=client).generate("prompt")

    assert result["title"] == "Title"
    assert client.responses.calls == 4
    assert sleeps == [1, 2, 4]


def test_openai_timeout_error_uses_existing_retry_mechanism(monkeypatch):
    timeout_error = APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))

    class TimeoutOnceResponses(FakeOpenAIResponses):
        def create(self, **kwargs):
            self.calls += 1
            self.kwargs.append(kwargs)
            if self.calls == 1:
                raise timeout_error
            return type("Response", (), {"output_text": self.content})()

    client = FakeOpenAIClient()
    client.responses = TimeoutOnceResponses(0, client.responses.content)
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    result = llm.NotificationGenerator(client=client).generate(
        "prompt",
        user_id=953,
        notification_type="SENTIMENT_QA",
    )

    assert result["title"] == "Title"
    assert client.responses.calls == 2
    assert sleeps == [1]


def test_openai_final_failure_logs_context_without_sensitive_data(monkeypatch, caplog):
    client = FakeOpenAIClient(failures=4)
    monkeypatch.setattr(llm, "_is_temporary_openai_error", lambda error: True)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)

    with caplog.at_level("ERROR", logger="app.llm.llm_client"):
        with pytest.raises(RuntimeError):
            llm.NotificationGenerator(client=client, model="test-model").generate(
                "secret prompt should not be logged",
                user_id=953,
                notification_type="SENTIMENT_QA",
            )

    message = caplog.text
    assert "OpenAI LLM failed after all retries" in message
    assert "user_id=953" in message
    assert "notification_type=SENTIMENT_QA" in message
    assert "model=test-model" in message
    assert "retry_attempt=4" in message
    assert "error_type=RuntimeError" in message
    assert "secret prompt should not be logged" not in message


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


@pytest.mark.parametrize(
    "notification_type, video_popup",
    [
        ("VIDEO_RECOMMENDATION", "Y"),
        ("SENTIMENT_ENGAGEMENT", "N"),
        ("SENTIMENT_QA", False),
    ],
)
def test_sender_payload_uses_only_backend_fields_and_boolean_popup(
    monkeypatch,
    notification_type,
    video_popup,
):
    captured = {}

    def post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("app.notifications.sender.requests.post", post)

    NotificationSender(remote_url="https://example.test").send(
        user_id=953,
        notification_type=notification_type,
        title="A notification",
        description="Notification details",
        reference_id=363,
        video_popup=video_popup,
    )

    payload = captured["json"]
    assert isinstance(payload, list)
    assert len(payload) == 1
    notification = payload[0]
    assert isinstance(notification, dict)
    assert set(notification) == {
        "user_id",
        "title",
        "description",
        "notification_type",
        "reference_id",
        "video_popup",
        "image",
    }
    assert notification["notification_type"] == notification_type
    assert isinstance(notification["video_popup"], bool)
    assert notification["video_popup"] is (notification_type == "VIDEO_RECOMMENDATION")
    assert isinstance(notification["reference_id"], int)
    assert notification["image"] is None


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


def test_sender_retry_eventually_succeeds_and_is_final_success(monkeypatch):
    attempts = []
    responses = [
        requests.Timeout("timeout"),
        FakeResponse(status_code=200, body={"ok": True}),
    ]

    def post(*args, **kwargs):
        attempts.append(1)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("app.notifications.sender.requests.post", post)
    monkeypatch.setattr("app.notifications.sender.time.sleep", lambda _: None)

    result = NotificationSender(remote_url="https://example.test").send(
        user_id=2438,
        notification_type="SENTIMENT_QA",
        title="Title",
        description="Description",
        reference_id=0,
    )

    assert result["status_code"] == 200
    assert result["request_payload"][0]["notification_type"] == "SENTIMENT_QA"
    assert len(attempts) == 2


def test_scheduler_skip_status_is_not_logged_as_failed_and_retry_loggers_are_silenced(caplog):
    run_daily.configure_terminal_logging()

    assert logging.getLogger("urllib3").level >= logging.ERROR
    assert logging.getLogger("urllib3.connectionpool").level >= logging.ERROR
    assert logging.getLogger("requests.packages.urllib3").level >= logging.ERROR

    with caplog.at_level("INFO", logger=run_daily.logger.name):
        run_daily._log_user_status(
            position=1,
            total_users=1,
            user_id=2526,
            user_name="Karthik",
            notification_type="VIDEO_RECOMMENDATION",
            status="SKIPPED",
        )

    output = caplog.text
    assert "STATUS : SKIPPED" in output
    assert "| SKIPPED" in output
    assert "| FAILED" not in output


def test_started_status_does_not_emit_a_final_remote_outcome(caplog):
    with caplog.at_level("INFO", logger=run_daily.logger.name):
        run_daily._log_user_status(
            position=1, total_users=1, user_id=7, user_name="User 7",
            notification_type="SENTIMENT_QA", status="STARTED",
        )

    assert "STATUS : STARTED" in caplog.text
    assert "[REMOTE]" not in caplog.text
    assert "SKIPPED" not in caplog.text
    assert "FAILED" not in caplog.text


def test_retry_library_terminal_logger_is_suppressed(caplog):
    run_daily.configure_terminal_logging()
    retry_logger = logging.getLogger("openai._base_client")
    retry_logger.info("Retrying request to https://example.test in 0.491939 seconds")

    assert "Retrying request in" not in caplog.text


def test_scheduler_emits_one_final_outcome_and_summary_counts(monkeypatch, caplog):
    engine = FakeSchedulerEngine([11, 12, 13, 14])

    class Result:
        notification_type = "SENTIMENT_QA"
        error = None
        reason = None

        def __init__(self, status, code=None):
            self.remote_send_status = status
            self.remote_send_response = {"status_code": code} if code is not None else None

    class Service:
        last_skip_reason = "NO_ENGAGEMENT_DATA"

        def build_notification(self, request):
            if request.user_id == 11:
                return Result("sent", 200)
            if request.user_id == 12:
                return Result("failed", 503)
            if request.user_id == 14:
                return Result("skipped")
            return None

    monkeypatch.setattr(run_daily, "engine", engine)
    monkeypatch.setattr(run_daily, "NotificationService", Service)
    monkeypatch.setattr(run_daily, "get_next_notification_for_user", lambda *a, **k: "SENTIMENT_QA")
    monkeypatch.setattr(run_daily, "has_notification_for_user_on_date", lambda *a, **k: False)
    with caplog.at_level("INFO", logger=run_daily.logger.name):
        summary = asyncio.run(run_daily.run_scheduler_async(test_mode=True))

    final_status_lines = [line for line in caplog.text.splitlines() if "STATUS : " in line and "STARTED" not in line]
    assert sum("SUCCESS" in line for line in final_status_lines) == 1
    assert sum("FAILED" in line for line in final_status_lines) == 2
    assert sum("SKIPPED" in line for line in final_status_lines) == 1
    assert summary["total_users"] == 4
    assert summary["successful_count"] == 1
    assert summary["failed_count"] == 2
    assert summary["skipped_count"] == 1


def test_scheduler_counts_final_outcome_after_retry_success_and_final_failure(monkeypatch):
    engine = FakeSchedulerEngine([1, 2, 3])

    class FinalSuccessResult:
        def __init__(self):
            self.notification_type = "SENTIMENT_QA"
            self.remote_send_status = "sent"
            self.remote_send_response = {"status_code": 200}
            self.error = None
            self.reason = None

    class FinalFailureResult:
        def __init__(self):
            self.notification_type = "SENTIMENT_QA"
            self.remote_send_status = "failed"
            self.remote_send_response = {"status_code": 500}
            self.error = "remote permanently failed"
            self.reason = "remote permanently failed"

    class FakeService:
        last_skip_reason = None

        def build_notification(self, request):
            if request.user_id == 1:
                return FinalSuccessResult()
            if request.user_id == 2:
                return FinalFailureResult()
            return None

    monkeypatch.setattr(run_daily, "engine", engine)
    monkeypatch.setattr(run_daily, "NotificationService", lambda: FakeService())
    monkeypatch.setattr(run_daily, "get_next_notification_for_user", lambda *args, **kwargs: "SENTIMENT_QA")
    monkeypatch.setattr(run_daily, "has_notification_for_user_on_date", lambda *args, **kwargs: False)

    summary = asyncio.run(run_daily.run_scheduler_async(test_mode=True))

    assert summary["total_users"] == 3
    assert summary["successful_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["skipped_count"] == 1


class FakeScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class FakeUserResult:
    def __init__(self, user_ids):
        self.user_ids = user_ids

    def __iter__(self):
        return iter([(user_id, f"User {user_id}") for user_id in self.user_ids])


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


def test_scheduler_selects_only_required_user_scope(monkeypatch):
    engine = FakeSchedulerEngine([])
    monkeypatch.setattr(run_daily, "engine", engine)
    monkeypatch.setattr(run_daily, "NotificationService", lambda: object())

    run_daily.main()

    selection_query = next(
        query
        for query in engine.connection.queries
        if "SELECT id" in query
    )
    normalized_query = " ".join(selection_query.split()).lower()
    assert 'from public."user"' in normalized_query
    assert "where account_id = 14" in normalized_query
    assert "and status = 1" in normalized_query
    assert "and debug = false" in normalized_query
    assert "and user_type_id = 1" in normalized_query
    assert "id is not null" not in normalized_query
    assert "coalesce" in normalized_query
    assert "name" in normalized_query


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
        title = "Title"
        description = "Body"
        reference_id = 10

        def model_dump(self, exclude_none=True):
            return {"notification_type": "VIDEO_RECOMMENDATION"}

    class FakeService:
        last_skip_reason = "NO_ENGAGEMENT_DATA"

        def build_notification(self, request):
            processed.append(request.user_id)
            if request.user_id == 2:
                raise RuntimeError("user 2 failed")
            return FakeResult()

    monkeypatch.setattr(run_daily, "engine", engine)
    monkeypatch.setattr(run_daily, "NotificationService", lambda: FakeService())
    monkeypatch.setattr(run_daily, "get_next_notification_for_user", lambda *args, **kwargs: "VIDEO_RECOMMENDATION")
    monkeypatch.setattr(run_daily, "has_notification_for_user_on_date", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        run_daily,
        "insert_notification",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("scheduler must persist through the notification URL")
        ),
        raising=False,
    )

    run_daily.main()

    assert processed == [1, 2, 3]
    assert any("pg_advisory_unlock" in query for query in engine.connection.queries)


def test_scheduler_logs_progress_names_types_and_skip_reasons(monkeypatch, caplog):
    engine = FakeSchedulerEngine([1, 2, 3, 4])
    notification_types = {
        1: "VIDEO_RECOMMENDATION",
        2: "SENTIMENT_ENGAGEMENT",
        3: "SENTIMENT_QA",
    }

    class FakeResult:
        remote_send_status = "sent"
        error = None

        def __init__(self, notification_type):
            self.notification_type = notification_type
            self.remote_send_response = {"status_code": 200}

    class FakeService:
        last_skip_reason = "NO_ENGAGEMENT_DATA"

        def build_notification(self, request):
            if request.user_id == 4:
                return None
            return FakeResult(notification_types[request.user_id])

    monkeypatch.setattr(run_daily, "engine", engine)
    monkeypatch.setattr(run_daily, "NotificationService", lambda: FakeService())
    monkeypatch.setattr(
        run_daily,
        "get_next_notification_for_user",
        lambda user_id, **kwargs: "VIDEO_RECOMMENDATION",
    )
    monkeypatch.setattr(
        run_daily,
        "has_notification_for_user_on_date",
        lambda user_id, event_type, **kwargs: False,
    )

    with caplog.at_level("INFO", logger=run_daily.logger.name):
        run_daily.main()

    output = caplog.text
    assert "[1/4] USER 1 | User 1" in output
    assert "[2/4] USER 2 | User 2" in output
    assert "[3/4] USER 3 | User 3" in output
    assert "[4/4] USER 4 | User 4" in output
    assert "TYPE   : VIDEO_RECOMMENDATION" in output
    assert "TYPE   : SENTIMENT_ENGAGEMENT" in output
    assert "TYPE   : SENTIMENT_QA" in output
    assert "STATUS : SUCCESS" in output
    assert "STATUS : SKIPPED" in output
    assert "[REMOTE] USER 1 | HTTP 200 | SUCCESS" in output
    assert "[JOB] SUMMARY" in output
    assert "[JOB] Total Users : 4" in output
    assert "[JOB] Successful  : 3" in output
    assert "[JOB] Failed      : 0" in output
    assert "[JOB] Skipped     : 1" in output
    assert "[JOB] COMPLETED" in output
    assert "[JOB] Eligible users got the notification" not in output


def test_scheduler_test_mode_parses_string_values_and_daily_limit_semantics(monkeypatch):
    assert run_daily.parse_test_mode("true") is True
    assert run_daily.parse_test_mode("false") is False
    assert run_daily.parse_test_mode(True) is True
    assert run_daily.parse_test_mode(False) is False
    assert run_daily.parse_test_mode("1") is True
    assert run_daily.parse_test_mode("0") is False

    monkeypatch.setenv("SCHEDULER_TEST_MODE", "true")
    assert run_daily.get_scheduler_test_mode() is True

    monkeypatch.setenv("SCHEDULER_TEST_MODE", "false")
    assert run_daily.get_scheduler_test_mode() is False

    monkeypatch.delenv("SCHEDULER_TEST_MODE", raising=False)


def test_scheduler_test_mode_bypasses_only_duplicate_check(monkeypatch, caplog):
    engine = FakeSchedulerEngine([953])
    processed = []

    class FakeResult:
        remote_send_status = "sent"
        notification_type = "SENTIMENT_QA"

    class FakeService:
        def build_notification(self, request):
            processed.append(request.user_id)
            return FakeResult()

    monkeypatch.setenv("SCHEDULER_TEST_MODE", "true")
    monkeypatch.setattr(run_daily, "engine", engine)
    monkeypatch.setattr(run_daily, "NotificationService", lambda: FakeService())
    monkeypatch.setattr(
        run_daily,
        "get_next_notification_for_user",
        lambda *args, **kwargs: "SENTIMENT_QA",
    )
    monkeypatch.setattr(
        run_daily,
        "has_notification_for_user_on_date",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate check must be bypassed in test mode")
        ),
    )

    with caplog.at_level("INFO", logger=run_daily.logger.name):
        run_daily.main()

    assert processed == [953]
    assert "[JOB] TEST MODE: ENABLED" in caplog.text


def test_scheduler_normal_mode_keeps_duplicate_skip(monkeypatch):
    engine = FakeSchedulerEngine([953])
    processed = []

    class FakeService:
        def build_notification(self, request):
            processed.append(request.user_id)
            raise AssertionError("generation must not run for duplicate notifications")

    monkeypatch.delenv("SCHEDULER_TEST_MODE", raising=False)
    monkeypatch.setattr(run_daily, "engine", engine)
    monkeypatch.setattr(run_daily, "NotificationService", lambda: FakeService())
    monkeypatch.setattr(
        run_daily,
        "get_next_notification_for_user",
        lambda *args, **kwargs: "SENTIMENT_QA",
    )
    monkeypatch.setattr(
        run_daily,
        "has_notification_for_user_on_date",
        lambda *args, **kwargs: True,
    )

    run_daily.main()

    assert processed == []


def test_scheduler_skips_sentiment_when_no_eligible_qa(monkeypatch):
    engine = FakeSchedulerEngine([953])
    requests_seen = []

    class FakeService:
        def build_notification(self, request):
            requests_seen.append(request)
            return None

    monkeypatch.setattr(run_daily, "engine", engine)
    monkeypatch.setattr(run_daily, "NotificationService", lambda: FakeService())
    monkeypatch.setattr(run_daily, "get_next_notification_for_user", lambda *args, **kwargs: "SENTIMENT_QA")
    monkeypatch.setattr(run_daily, "has_notification_for_user_on_date", lambda *args, **kwargs: False)

    run_daily.main()

    assert requests_seen[0].flow == "sentiment"


def test_scheduler_uses_default_fallback_without_manual_flag(monkeypatch):
    class FakeResult:
        def __init__(self):
            self.remote_send_status = "sent"
            self.remote_send_response = {"status_code": 200}
            self.notification_type = "VIDEO_RECOMMENDATION"
            self.reason = "NO_VIDEO_RECOMMENDATION"
            self.error = None

    class FakeService:
        last_skip_reason = "NO_VIDEO_RECOMMENDATION"

        async def build_notification_async(self, request, **kwargs):
            assert "allow_fallback" not in kwargs
            assert request.flow == "performance"
            return FakeResult()

    monkeypatch.setattr(run_daily, "NotificationService", lambda: FakeService())
    monkeypatch.setattr(run_daily, "get_next_notification_for_user", lambda *args, **kwargs: "VIDEO_RECOMMENDATION")
    monkeypatch.setattr(run_daily, "has_notification_for_user_on_date", lambda *args, **kwargs: False)
    result = asyncio.run(
        run_daily._process_user_async(
            953,
            "Ava",
            position=1,
            total_users=1,
            test_mode=False,
            semaphore=asyncio.Semaphore(5),
        )
    )

    assert result["status"] == "success"
    assert result["user_id"] == 953


def test_openai_async_generate_uses_awaited_client_call(monkeypatch):
    class AsyncFakeResponses:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            return type("Response", (), {"output_text": json.dumps({
                "title": "Title",
                "description": "Description",
                "action": "Watch now",
            })})()

    class AsyncFakeOpenAIClient:
        def __init__(self):
            self.responses = AsyncFakeResponses()

    async def run_test():
        client = AsyncFakeOpenAIClient()
        result = await llm.NotificationGenerator(client=client).generate_async("prompt")
        assert result["title"] == "Title"
        assert client.responses.calls == 1

    asyncio.run(run_test())


def test_sender_async_send_uses_async_http_client(monkeypatch):
    class AsyncResponse:
        ok = True
        status_code = 200
        text = "ok"

        async def json(self):
            return {"ok": True}

    class AsyncClient:
        def __init__(self):
            self.calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return AsyncResponse()

    async def run_test():
        captured = {}
        monkeypatch.setattr("app.notifications.sender.httpx", type("HttpxModule", (), {"AsyncClient": lambda *a, **k: captured.setdefault("client", AsyncClient())}))

        result = await NotificationSender(remote_url="https://example.test").send_async(
            user_id=1,
            notification_type="VIDEO_RECOMMENDATION",
            title="Title",
            description="Description",
            reference_id=10,
        )

        assert result["status_code"] == 200
        assert captured["client"].calls

    asyncio.run(run_test())


def test_scheduler_limits_concurrency_to_five(monkeypatch):
    active = 0
    peak = 0

    async def fake_process_user(user_id, user_name, *, position=None, total_users=None, test_mode=None, semaphore=None):
        nonlocal active, peak
        async with semaphore:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
        return user_id

    async def run_test():
        monkeypatch.setattr(run_daily, "_process_user_async", fake_process_user)
        monkeypatch.setattr(run_daily, "_collect_users", lambda: [(i, f"User {i}") for i in range(1, 26)])

        result = await run_daily.run_scheduler_async()

        assert result["total_users"] == 25
        assert peak <= 5

    asyncio.run(run_test())
