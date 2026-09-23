from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from app.database.notification_repository import (
    determine_user_cycle_day,
    get_next_manual_notification_for_user,
    get_next_notification_for_user,
    has_notification_for_user_on_date,
    schedule_next_notification_for_user,
)


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class FakeConnection:
    def __init__(self, rows_by_user):
        self.rows_by_user = rows_by_user

    def execute(self, query, params=None):
        q = str(query).upper()
        if "INSERT INTO PUBLIC.NOTIFICATION" in q:
            user_id = int(params["target_user_id"])
            self.rows_by_user.setdefault(user_id, []).append({
                "event_type": params["event_type"],
                "created_at": params.get("created_at", datetime.now(timezone.utc)),
                "status": 1,
            })
            return FakeResult([])

        user_id = params.get("user_id") if params else None
        rows = self.rows_by_user.get(user_id, [])
        if "ORDER BY ID DESC" in q:
            rows = sorted(rows, key=lambda row: row.get("id", 0), reverse=True)
        if params and "created_on" in params and rows:
            rows = [
                row for row in rows
                if row.get("event_type") == params["event_type"]
                and row.get("status", 1) == 1
                and row.get("created_at").astimezone(ZoneInfo("Asia/Kolkata")).date() == params["created_on"]
            ]
        return FakeResult(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def begin(self):
        return self


class FakeEngine:
    def __init__(self, rows_by_user=None):
        self.rows_by_user = rows_by_user or {}

    def connect(self):
        return FakeConnection(self.rows_by_user)

    def begin(self):
        return FakeConnection(self.rows_by_user)


def test_notification_at_2330_utc_is_current_ist_day():
    engine = FakeEngine({953: [{
        "event_type": "SENTIMENT_QA",
        "created_at": datetime(2026, 9, 22, 23, 30, tzinfo=timezone.utc),
    }]})

    assert has_notification_for_user_on_date(
        953,
        "SENTIMENT_QA",
        db_engine=engine,
        as_of_date=date(2026, 9, 23),
    ) is True


def test_notification_at_0030_utc_is_current_ist_day():
    engine = FakeEngine({953: [{
        "event_type": "SENTIMENT_QA",
        "created_at": datetime(2026, 9, 22, 0, 30, tzinfo=timezone.utc),
    }]})

    assert has_notification_for_user_on_date(
        953,
        "SENTIMENT_QA",
        db_engine=engine,
        as_of_date=date(2026, 9, 22),
    ) is True


def test_notification_from_previous_ist_day_is_not_current():
    engine = FakeEngine({953: [{
        "event_type": "SENTIMENT_QA",
        "created_at": datetime(2026, 9, 21, 17, 0, tzinfo=timezone.utc),
    }]})

    assert has_notification_for_user_on_date(
        953,
        "SENTIMENT_QA",
        db_engine=engine,
        as_of_date=date(2026, 9, 22),
    ) is False


def test_current_active_notification_matches_same_user_and_event_type():
    engine = FakeEngine({953: [{
        "event_type": "SENTIMENT_ENGAGEMENT",
        "status": 1,
        "created_at": datetime(2026, 9, 22, 3, 0, tzinfo=timezone.utc),
    }]})

    assert has_notification_for_user_on_date(
        953,
        "SENTIMENT_ENGAGEMENT",
        db_engine=engine,
        as_of_date=date(2026, 9, 22),
    ) is True


def test_different_user_or_event_type_does_not_match():
    engine = FakeEngine({953: [{
        "event_type": "SENTIMENT_ENGAGEMENT",
        "status": 1,
        "created_at": datetime(2026, 9, 22, 3, 0, tzinfo=timezone.utc),
    }]})

    assert has_notification_for_user_on_date(
        954,
        "SENTIMENT_ENGAGEMENT",
        db_engine=engine,
        as_of_date=date(2026, 9, 22),
    ) is False
    assert has_notification_for_user_on_date(
        953,
        "SENTIMENT_QA",
        db_engine=engine,
        as_of_date=date(2026, 9, 22),
    ) is False


def test_day_1_user_cycle():
    engine = FakeEngine({})
    assert determine_user_cycle_day(101, db_engine=engine, as_of_date=date(2026, 1, 1)) == 1
    assert get_next_notification_for_user(101, db_engine=engine, as_of_date=date(2026, 1, 1)) == "VIDEO_RECOMMENDATION"


def test_day_2_cycle():
    engine = FakeEngine({101: [{"event_type": "VIDEO_RECOMMENDATION", "created_at": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)}]})
    assert determine_user_cycle_day(101, db_engine=engine, as_of_date=date(2026, 1, 2)) == 2
    assert get_next_notification_for_user(101, db_engine=engine, as_of_date=date(2026, 1, 2)) == "SENTIMENT_ENGAGEMENT"


def test_day_3_cycle():
    engine = FakeEngine({101: [{"event_type": "VIDEO_RECOMMENDATION", "created_at": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)}]})
    assert determine_user_cycle_day(101, db_engine=engine, as_of_date=date(2026, 1, 3)) == 3
    assert get_next_notification_for_user(101, db_engine=engine, as_of_date=date(2026, 1, 3)) == "SENTIMENT_QA"


def test_day_4_cycle():
    engine = FakeEngine({101: [{"event_type": "VIDEO_RECOMMENDATION", "created_at": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)}]})
    assert determine_user_cycle_day(101, db_engine=engine, as_of_date=date(2026, 1, 4)) == 4
    assert get_next_notification_for_user(101, db_engine=engine, as_of_date=date(2026, 1, 4)) == "VIDEO_RECOMMENDATION"


def test_day_5_cycle():
    engine = FakeEngine({101: [{"event_type": "VIDEO_RECOMMENDATION", "created_at": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)}]})
    assert determine_user_cycle_day(101, db_engine=engine, as_of_date=date(2026, 1, 5)) == 5
    assert get_next_notification_for_user(101, db_engine=engine, as_of_date=date(2026, 1, 5)) == "SENTIMENT_ENGAGEMENT"


def test_day_6_cycle():
    engine = FakeEngine({101: [{"event_type": "VIDEO_RECOMMENDATION", "created_at": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)}]})
    assert determine_user_cycle_day(101, db_engine=engine, as_of_date=date(2026, 1, 6)) == 6
    assert get_next_notification_for_user(101, db_engine=engine, as_of_date=date(2026, 1, 6)) == "SENTIMENT_QA"


def test_day_7_is_break_and_day_8_resets_to_video_recommendation():
    engine = FakeEngine({101: [{"event_type": "VIDEO_RECOMMENDATION", "created_at": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)}]})
    assert determine_user_cycle_day(101, db_engine=engine, as_of_date=date(2026, 1, 7)) == 7
    assert get_next_notification_for_user(101, db_engine=engine, as_of_date=date(2026, 1, 7)) is None
    assert determine_user_cycle_day(101, db_engine=engine, as_of_date=date(2026, 1, 8)) == 1
    assert get_next_notification_for_user(101, db_engine=engine, as_of_date=date(2026, 1, 8)) == "VIDEO_RECOMMENDATION"


def test_new_user_starts_on_day_1():
    engine = FakeEngine({})
    assert determine_user_cycle_day(999, db_engine=engine, as_of_date=date(2026, 2, 1)) == 1
    assert get_next_notification_for_user(999, db_engine=engine, as_of_date=date(2026, 2, 1)) == "VIDEO_RECOMMENDATION"


def test_no_event_starts_cycle_at_first_valid_notification():
    engine = FakeEngine({})

    assert determine_user_cycle_day(555, db_engine=engine, as_of_date=date(2026, 2, 1)) == 1
    assert get_next_notification_for_user(555, db_engine=engine, as_of_date=date(2026, 2, 1)) == "VIDEO_RECOMMENDATION"

    reset_engine = FakeEngine({
        101: [{"event_type": "VIDEO_RECOMMENDATION", "created_at": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)}],
    })
    assert get_next_notification_for_user(101, db_engine=reset_engine, as_of_date=date(2026, 1, 7)) is None
    assert determine_user_cycle_day(101, db_engine=reset_engine, as_of_date=date(2026, 1, 8)) == 1
    assert get_next_notification_for_user(101, db_engine=reset_engine, as_of_date=date(2026, 1, 8)) == "VIDEO_RECOMMENDATION"


def test_manual_cycle_uses_latest_event_type():
    for event_type, expected in [
        ("VIDEO_RECOMMENDATION", "SENTIMENT_ENGAGEMENT"),
        ("SENTIMENT_ENGAGEMENT", "SENTIMENT_QA"),
        ("SENTIMENT_QA", "VIDEO_RECOMMENDATION"),
    ]:
        engine = FakeEngine({953: [{"event_type": event_type, "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}]})
        assert get_next_manual_notification_for_user(953, db_engine=engine) == expected


def test_manual_cycle_uses_highest_notification_id():
    engine = FakeEngine({953: [
        {"id": 10, "event_type": "VIDEO_RECOMMENDATION", "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        {"id": 11, "event_type": "SENTIMENT_ENGAGEMENT", "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc)},
    ]})

    assert get_next_manual_notification_for_user(953, db_engine=engine) == "SENTIMENT_QA"


def test_manual_cycle_starts_with_performance_for_new_user():
    assert get_next_manual_notification_for_user(953, db_engine=FakeEngine()) == "VIDEO_RECOMMENDATION"


def test_users_have_independent_cycles():
    engine = FakeEngine({
        101: [{"event_type": "VIDEO_RECOMMENDATION", "created_at": datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)}],
        202: [{"event_type": "VIDEO_RECOMMENDATION", "created_at": datetime(2026, 3, 3, 9, 0, tzinfo=timezone.utc)}],
    })
    assert determine_user_cycle_day(101, db_engine=engine, as_of_date=date(2026, 3, 5)) == 5
    assert determine_user_cycle_day(202, db_engine=engine, as_of_date=date(2026, 3, 5)) == 3
    assert get_next_notification_for_user(101, db_engine=engine, as_of_date=date(2026, 3, 5)) == "SENTIMENT_ENGAGEMENT"
    assert get_next_notification_for_user(202, db_engine=engine, as_of_date=date(2026, 3, 5)) == "SENTIMENT_QA"


def test_running_scheduler_twice_does_not_create_duplicates():
    stored_by_user = {}

    class RecordingEngine:
        def connect(self):
            return FakeConnection(stored_by_user)

        def begin(self):
            return FakeConnection(stored_by_user)

    engine = RecordingEngine()

    first = schedule_next_notification_for_user(
        user_id=555,
        title="title",
        description="body",
        db_engine=engine,
        as_of_date=date(2026, 4, 1),
        event_details={"source": "scheduler"},
    )
    second = schedule_next_notification_for_user(
        user_id=555,
        title="title",
        description="body",
        db_engine=engine,
        as_of_date=date(2026, 4, 1),
        event_details={"source": "scheduler"},
    )

    assert first is not None
    assert second is None
    assert first["event_type"] == "VIDEO_RECOMMENDATION"
