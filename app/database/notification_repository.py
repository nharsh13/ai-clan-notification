from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import bindparam, text

from app.constants import NEXT_NOTIFICATION_BY_EVENT_TYPE
from app.database.connection import engine


def get_user_notification_history(user_id: int, db_engine=engine) -> list[dict[str, Any]]:
    query = text(
        """
        SELECT
            id,
            target_user_id,
            event_type,
            event_ref_id,
            event_details,
            created_at,
            status
        FROM public.notification
        WHERE target_user_id = :user_id
          AND status = 1
        ORDER BY created_at ASC, id ASC
        """
    )

    with db_engine.connect() as connection:
        rows = connection.execute(query, {"user_id": user_id}).mappings().all()

    return [dict(row) for row in rows]


def get_next_notification_for_user(user_id: int, *, db_engine=engine, as_of_date: date | None = None) -> str | None:
    # Keep as_of_date for callers that still pass it; event sequencing is history-based.
    return get_next_manual_notification_for_user(user_id, db_engine=db_engine)


def get_next_manual_notification_for_user(user_id: int, *, db_engine=engine) -> str:
    query = text(
        """
        SELECT event_type
        FROM public.notification
        WHERE target_user_id = :user_id
          AND status = 1
          AND event_type IN :valid_event_types
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """
    ).bindparams(bindparam("valid_event_types", expanding=True))

    with db_engine.connect() as connection:
        row = connection.execute(query, {
            "user_id": user_id,
            "valid_event_types": tuple(NEXT_NOTIFICATION_BY_EVENT_TYPE),
        }).mappings().first()

    if row is None:
        return "VIDEO_RECOMMENDATION"

    return NEXT_NOTIFICATION_BY_EVENT_TYPE[row["event_type"]]


def has_notification_for_user_on_date(
    user_id: int,
    event_type: str,
    *,
    db_engine=engine,
    as_of_date: date | None = None,
) -> bool:
    query = text(
        """
        SELECT 1
        FROM public.notification
        WHERE target_user_id = :user_id
          AND event_type = :event_type
          AND status = 1
          AND (created_at AT TIME ZONE 'Asia/Kolkata')::date = :created_on
        LIMIT 1
        """
    )

    created_on = as_of_date or datetime.now(ZoneInfo("Asia/Kolkata")).date()
    with db_engine.connect() as connection:
        row = connection.execute(query, {
            "user_id": user_id,
            "event_type": event_type,
            "created_on": created_on,
        }).first()

    return row is not None


def schedule_next_notification_for_user(
    *,
    user_id: int,
    title: str,
    description: str,
    db_engine=engine,
    as_of_date: date | None = None,
    event_ref_id: int | None = None,
    event_details: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    event_type = get_next_notification_for_user(user_id, db_engine=db_engine, as_of_date=as_of_date)
    if event_type is None:
        return None
    if has_notification_for_user_on_date(user_id, event_type, db_engine=db_engine, as_of_date=as_of_date):
        return None

    scheduled_at = datetime.combine(as_of_date or datetime.now(timezone.utc).date(), datetime.min.time(), tzinfo=timezone.utc)
    return insert_notification(
        target_user_id=user_id,
        event_type=event_type,
        title=title,
        description=description,
        event_ref_id=event_ref_id,
        event_details=event_details,
        created_at=scheduled_at,
        db_engine=db_engine,
    )


def insert_notification(
    *,
    target_user_id: int,
    event_type: str,
    title: str,
    description: str,
    event_ref_id: int | None = None,
    event_details: dict[str, Any] | None = None,
    created_at: datetime | None = None,
    db_engine=engine,
) -> dict[str, Any]:
    created_at = created_at or datetime.now(timezone.utc)
    payload = event_details or {}
    query = text(
        """
        INSERT INTO public.notification (
            title,
            description,
            notification_read,
            created_at,
            updated_at,
            target_user_id,
            event_initiator_id,
            event_type,
            event_ref_id,
            created_by,
            modified_by,
            status,
            event_details
        ) VALUES (
            :title,
            :description,
            :notification_read,
            :created_at,
            :updated_at,
            :target_user_id,
            :event_initiator_id,
            :event_type,
            :event_ref_id,
            :created_by,
            :modified_by,
            :status,
            :event_details
        )
        """
    )

    with db_engine.begin() as connection:
        connection.execute(
            query,
            {
                "title": title,
                "description": description,
                "notification_read": False,
                "created_at": created_at,
                "updated_at": created_at,
                "target_user_id": target_user_id,
                "event_initiator_id": target_user_id,
                "event_type": event_type,
                "event_ref_id": event_ref_id,
                "created_by": target_user_id,
                "modified_by": target_user_id,
                "status": 1,
                "event_details": payload,
            },
        )

    return {
        "target_user_id": target_user_id,
        "event_type": event_type,
        "title": title,
        "description": description,
        "event_ref_id": event_ref_id,
        "event_details": payload,
    }
