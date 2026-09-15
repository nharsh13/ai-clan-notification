from datetime import datetime, timedelta

import app.notifications.service as service_module
from app.notifications.service import NotificationService
from app.sentiment import sentiment


class FakeResult:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


class SelectionConnection:
    def __init__(self, responses, history, today):
        self.responses = responses
        self.history = history
        self.today = today

    def execute(self, query, params):
        assert "r.status = 1" in str(query)
        assert "sentiment_notification_history" in str(query)
        active_history = [row for row in self.history if row["status"] == 1]
        if any(row["user_id"] == params["user_id"] and row["created_at"].date() == self.today for row in active_history):
            return FakeResult(None)

        used_ids = {
            row["response_id"]
            for row in active_history
            if row["user_id"] == params["user_id"]
        }
        eligible = [
            row for row in self.responses
            if row["user_id"] == params["user_id"]
            and row["status"] == 1
            and row["response_id"] not in used_ids
        ]
        eligible.sort(key=lambda row: (row["created_at"], row["response_id"]))
        return FakeResult(eligible[0] if eligible else None)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class SelectionEngine:
    def __init__(self, responses, history, today):
        self.connection = SelectionConnection(responses, history, today)

    def connect(self):
        return self.connection


class WriteConnection:
    def __init__(self):
        self.query = None
        self.params = None

    def execute(self, query, params):
        self.query = str(query)
        self.params = params

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class WriteEngine:
    def __init__(self):
        self.connection = WriteConnection()

    def begin(self):
        return self.connection


def response(response_id, question_id, answer_id, created_at, status=1):
    return {
        "response_id": response_id,
        "user_id": 953,
        "question_id": question_id,
        "question": f"Question {question_id}",
        "answer_id": answer_id,
        "answer": f"Answer {answer_id}",
        "created_at": created_at,
        "status": status,
    }


def test_unanswered_response_is_ignored():
    now = datetime(2026, 9, 15, 8)
    result = sentiment.get_next_sentiment_response(
        953,
        SelectionEngine([response(1, 1, 1, now, status=0)], [], now.date()),
    )
    assert result is None


def test_answered_response_is_selected_chronologically():
    now = datetime(2026, 9, 15, 8)
    result = sentiment.get_next_sentiment_response(
        953,
        SelectionEngine(
            [response(2, 2, 2, now + timedelta(hours=1)), response(1, 1, 1, now)],
            [],
            now.date(),
        ),
    )
    assert result["response_id"] == 1
    assert set(result) >= {"response_id", "question", "answer"}


def test_processed_response_id_is_skipped():
    now = datetime(2026, 9, 15, 8)
    result = sentiment.get_next_sentiment_response(
        953,
        SelectionEngine(
            [response(1, 1, 1, now), response(2, 2, 2, now + timedelta(hours=1))],
            [{"user_id": 953, "response_id": 1, "created_at": now - timedelta(days=1), "status": 1}],
            now.date(),
        ),
    )
    assert result["response_id"] == 2


def test_only_one_sentiment_notification_per_day():
    now = datetime(2026, 9, 15, 8)
    result = sentiment.get_next_sentiment_response(
        953,
        SelectionEngine(
            [response(2, 2, 2, now + timedelta(hours=1))],
            [{"user_id": 953, "response_id": 1, "created_at": now, "status": 1}],
            now.date(),
        ),
    )
    assert result is None


def test_next_unused_response_is_selected_on_next_day():
    now = datetime(2026, 9, 16, 8)
    result = sentiment.get_next_sentiment_response(
        953,
        SelectionEngine(
            [response(1, 1, 1, now - timedelta(days=2)), response(2, 2, 2, now - timedelta(days=1))],
            [{"user_id": 953, "response_id": 1, "created_at": now - timedelta(days=1), "status": 1}],
            now.date(),
        ),
    )
    assert result["response_id"] == 2


def test_same_question_and_answer_with_new_response_id_is_eligible():
    now = datetime(2026, 9, 16, 8)
    result = sentiment.get_next_sentiment_response(
        953,
        SelectionEngine(
            [response(2, 1, 1, now)],
            [{"user_id": 953, "response_id": 1, "created_at": now - timedelta(days=1), "status": 1}],
            now.date(),
        ),
    )
    assert result["response_id"] == 2


def test_no_eligible_response_returns_no_notification(monkeypatch):
    monkeypatch.setattr(service_module, "get_next_sentiment_response", lambda *args: None)
    result = NotificationService().process_sentiment_notification(953)
    assert result == {"user_id": 953, "notification": None}


def test_history_insert_is_attempted_after_successful_send(monkeypatch):
    selected = response(1, 7, 8, datetime(2026, 9, 15, 8))
    saved = []

    class Sender:
        remote_url = "https://example.test/notify"

        def send(self, **kwargs):
            return {"status": "ok"}

    monkeypatch.setattr(service_module, "get_next_sentiment_response", lambda *args: selected)
    monkeypatch.setattr(
        service_module,
        "save_sentiment_notification_history",
        lambda response, db_engine: saved.append(response),
    )

    result = NotificationService(sender=Sender()).process_sentiment_notification(953)

    assert result["remote_send_status"] == "sent"
    assert saved == [selected]


def test_history_helper_inserts_required_response_identity():
    selected = response(12, 7, 8, datetime(2026, 9, 15, 8))
    db_engine = WriteEngine()

    sentiment.save_sentiment_notification_history(selected, db_engine)

    assert "INSERT INTO public.sentiment_notification_history" in db_engine.connection.query
    assert db_engine.connection.params == {
        "user_id": 953,
        "response_id": 12,
        "question_id": 7,
        "answer_id": 8,
    }