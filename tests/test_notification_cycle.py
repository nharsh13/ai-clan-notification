from __future__ import annotations

from datetime import datetime, timezone

from app.database.notification_repository import (
    get_next_manual_notification_for_user,
    get_next_notification_for_user,
    has_notification_for_user_on_date,
)


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None


class FakeConnection:
    def __init__(self, rows_by_user):
        self.rows_by_user = rows_by_user

    def execute(self, query, params=None):
        q = str(query).upper()
        if "VALID_EVENT_TYPES" in q:
            user_id = int(params["user_id"])
            valid = set(params["valid_event_types"])
            rows = [
                row for row in self.rows_by_user.get(user_id, [])
                if row.get("status", 1) == 1 and row.get("event_type") in valid
            ]
            rows = sorted(
                rows,
                key=lambda row: (row.get("created_at", datetime.now(timezone.utc)), row.get("id", 0)),
                reverse=True,
            )
            return FakeResult(rows)

        if "FROM PUBLIC.NOTIFICATION" in q and "EVENT_TYPE =" in q:
            user_id = int(params["user_id"])
            event_type = params["event_type"]
            rows = [
                row for row in self.rows_by_user.get(user_id, [])
                if row.get("status", 1) == 1 and row.get("event_type") == event_type
            ]
            return FakeResult(rows)

        return FakeResult([])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeEngine:
    def __init__(self, rows_by_user=None):
        self.rows_by_user = rows_by_user or {}

    def connect(self):
        return FakeConnection(self.rows_by_user)


def test_new_user_starts_with_video_recommendation():
    assert get_next_manual_notification_for_user(953, db_engine=FakeEngine()) == "VIDEO_RECOMMENDATION"


def test_active_sequence_rotates_after_each_event():
    cases = [
        ("VIDEO_RECOMMENDATION", "SENTIMENT_ENGAGEMENT"),
        ("SENTIMENT_ENGAGEMENT", "SENTIMENT_QA"),
        ("SENTIMENT_QA", "VIDEO_RECOMMENDATION"),
        ("VIDEO_RECOMMENDATION", "SENTIMENT_ENGAGEMENT"),
    ]

    for current, expected in cases:
        engine = FakeEngine({953: [{"event_type": current, "status": 1, "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}]})
        assert get_next_notification_for_user(953, db_engine=engine) == expected


def test_active_sequence_ignores_non_valid_events_and_inactive_rows():
    engine = FakeEngine({953: [
        {"id": 1, "event_type": "UNRELATED_EVENT", "status": 1, "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        {"id": 2, "event_type": "VIDEO_RECOMMENDATION", "status": 0, "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc)},
        {"id": 3, "event_type": "VIDEO_RECOMMENDATION", "status": 1, "created_at": datetime(2026, 1, 3, tzinfo=timezone.utc)},
    ]})

    assert get_next_manual_notification_for_user(953, db_engine=engine) == "SENTIMENT_ENGAGEMENT"


def test_duplicate_today_only_blocks_same_event_type_for_same_user():
    today = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
    engine = FakeEngine({953: [{"event_type": "VIDEO_RECOMMENDATION", "status": 1, "created_at": today}]})

    assert has_notification_for_user_on_date(953, "VIDEO_RECOMMENDATION", db_engine=engine) is True
    assert has_notification_for_user_on_date(953, "SENTIMENT_ENGAGEMENT", db_engine=engine) is False
    assert has_notification_for_user_on_date(954, "VIDEO_RECOMMENDATION", db_engine=engine) is False


def test_sequence_is_history_based_not_day_based():
    engine = FakeEngine({953: [{"event_type": "SENTIMENT_QA", "status": 1, "created_at": datetime(2026, 1, 15, tzinfo=timezone.utc)}]})

    assert get_next_manual_notification_for_user(953, db_engine=engine) == "VIDEO_RECOMMENDATION"
