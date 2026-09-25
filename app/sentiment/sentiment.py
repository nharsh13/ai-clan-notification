from datetime import datetime, timezone
import logging

from sqlalchemy import text

from app.database.connection import engine

logger = logging.getLogger(__name__)


def _select_next_question_for_cycle(question_rows: list[dict], used_question_ids: set[int]) -> dict | None:
    """Choose the next unused question, or reset to the oldest question when the cycle is exhausted."""
    eligible = [row for row in question_rows if int(row["question_id"]) not in used_question_ids]
    if not eligible:
        eligible = list(question_rows)
    if not eligible:
        return None

    def _sort_key(row: dict):
        created_at = row.get("first_created_at")
        if created_at is None:
            created_at = datetime.min.replace(tzinfo=timezone.utc)
        response_id = row.get("first_response_id") or 0
        return (created_at, response_id, int(row["question_id"]))

    return min(eligible, key=_sort_key)


def get_user_answered_questions(user_id: int, db_engine=engine):
    """
    Fetch all answered CLAN questions for a user.
    """

    query = text("""
        SELECT
            r.user_id,
            r.question_id,
            q.question AS question_text,
            r.answer_id,
            a.answer_text
        FROM public.user_persona_question_responces r
        JOIN public.user_persona_question q
            ON q.id = r.question_id
        JOIN public.user_persona_question_answers a
            ON a.id = r.answer_id
            AND a.question_id = r.question_id
        WHERE r.status = 1
          AND r.user_id = :user_id
        ORDER BY r.question_id;
    """)

    with db_engine.connect() as connection:
        result = connection.execute(
            query,
            {"user_id": user_id}
        )

        return [
            {
                "user_id": row.user_id,
                "question_id": row.question_id,
                "question": row.question_text,
                "answer_id": row.answer_id,
                "answer": row.answer_text,
            }
            for row in result
        ]


def prepare_user_qa(
    user_id: int,
    db_engine=engine,
    rows: list[dict] | None = None,
):
    """
    Group all answered Q&A for a user by question.
    """

    rows = rows if rows is not None else get_user_answered_questions(user_id, db_engine)

    grouped = {}

    for row in rows:
        grouped.setdefault(row["question_id"], []).append({
            "question": row["question"],
            "answer_id": row["answer_id"],
            "answer": row["answer"],
        })

    return {
        "user_id": user_id,
        "questions": [
            {
                "question_id": question_id,
                "responses": responses,
            }
            for question_id, responses in grouped.items()
        ],
    }


def get_eligible_user_qa(user_id: int, db_engine=engine) -> list[dict]:
    """Return the next unused question and all of its related answers for one notification."""

    question_query = text("""
        WITH answered_questions AS (
            SELECT
                r.question_id,
                q.question,
                MIN(r.created_at) AS first_created_at,
                MIN(r.id) AS first_response_id
            FROM public.user_persona_question_responces r
            JOIN public.user_persona_question q
                ON q.id = r.question_id
            WHERE r.user_id = :user_id
              AND r.status = 1
            GROUP BY r.question_id, q.question
        ), assigned_without_answers AS (
            SELECT
                a.question_id,
                q.question,
                MIN(a.created_at) AS first_created_at,
                NULL::bigint AS first_response_id
            FROM public.user_persona_question_assignment a
            JOIN public.user_persona_question q
                ON q.id = a.question_id
            WHERE a.user_id = :user_id
              AND a.status = 1
              AND NOT EXISTS (
                  SELECT 1
                  FROM public.user_persona_question_responces r
                  WHERE r.user_id = a.user_id
                    AND r.question_id = a.question_id
                    AND r.status = 1
              )
            GROUP BY a.question_id, q.question
        ), all_questions AS (
            SELECT * FROM answered_questions
            UNION ALL
            SELECT * FROM assigned_without_answers
        )
        SELECT question_id, question, first_created_at, first_response_id
        FROM all_questions
        ORDER BY first_created_at, first_response_id, question_id
    """)

    used_question_query = text("""
        SELECT DISTINCT h.question_id
        FROM public.sentiment_notification_history h
        WHERE h.user_id = :user_id
          AND h.status = 1
          AND h.question_id IS NOT NULL
    """)

    with db_engine.connect() as connection:
        candidate_rows = [dict(row) for row in connection.execute(question_query, {"user_id": user_id}).mappings()]
        used_question_ids = {
            int(row["question_id"])
            for row in connection.execute(used_question_query, {"user_id": user_id}).mappings()
        }

    selected_question = _select_next_question_for_cycle(candidate_rows, used_question_ids)
    if selected_question is None:
        return []

    candidate_ids = {int(row["question_id"]) for row in candidate_rows}
    selection_status = (
        "CYCLE_RESET"
        if candidate_ids and candidate_ids.issubset(used_question_ids)
        else "UNUSED"
    )
    candidate_ids = sorted(candidate_ids)
    unused_question_ids = sorted(set(candidate_ids) - used_question_ids)

    selected_question_id = int(selected_question["question_id"])
    responses_query = text("""
        SELECT
            r.id AS response_id,
            r.user_id,
            r.question_id,
            q.question,
            r.answer_id,
            a.answer_text AS answer,
            r.created_at
        FROM public.user_persona_question_responces r
        JOIN public.user_persona_question q
            ON q.id = r.question_id
        JOIN public.user_persona_question_answers a
            ON a.id = r.answer_id
           AND a.question_id = r.question_id
        WHERE r.user_id = :user_id
          AND r.question_id = :question_id
          AND r.status = 1
        ORDER BY r.created_at, r.id
    """)

    with db_engine.connect() as connection:
        response_rows = [dict(row) for row in connection.execute(responses_query, {"user_id": user_id, "question_id": selected_question_id}).mappings()]

    if not response_rows:
        return [{
            "user_id": user_id,
            "question_id": selected_question_id,
            "question": selected_question["question"],
            "answer_id": None,
            "answer": None,
            "selection_status": selection_status,
            "used_question_ids": sorted(used_question_ids),
            "unused_question_ids": unused_question_ids,
        }]

    for row in response_rows:
        row["selection_status"] = selection_status
        row["used_question_ids"] = sorted(used_question_ids)
        row["unused_question_ids"] = unused_question_ids
    return response_rows


def get_next_sentiment_response(user_id: int, db_engine=engine) -> dict | None:
    rows = get_eligible_user_qa(user_id, db_engine)
    if not rows:
        return None

    question_id = rows[0]["question_id"]
    selected = {
        "user_id": user_id,
        "question_id": question_id,
        "question": rows[0]["question"],
        "responses": rows,
    }
    for row in rows:
        selected.setdefault("response_id", row.get("response_id"))
        selected.setdefault("answer_id", row.get("answer_id"))
        selected.setdefault("answer", row.get("answer"))
    return selected


def save_sentiment_notification_history(
    response: dict | list[dict],
    db_engine=engine,
) -> None:
    query = text("""
        INSERT INTO public.sentiment_notification_history (
            user_id, response_id, question_id, answer_id,
            created_at, modified_at, created_by, modified_by, status
        ) VALUES (
            :user_id, :response_id, :question_id, :answer_id,
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, :user_id, :user_id, 1
        )
    """)

    responses = response if isinstance(response, list) else [response]
    valid_responses = [
        item for item in responses
        if item.get("user_id") is not None
        and item.get("question_id") is not None
    ]
    if not valid_responses:
        return

    with db_engine.begin() as connection:
        reset_user_ids = {
            item["user_id"]
            for item in valid_responses
            if item.get("selection_status") == "CYCLE_RESET"
        }
        if reset_user_ids:
            reset_query = text("""
                UPDATE public.sentiment_notification_history
                SET status = 0,
                    modified_at = CURRENT_TIMESTAMP,
                    modified_by = :user_id
                WHERE user_id = :user_id
                  AND status = 1
            """)
            for user_id in reset_user_ids:
                connection.execute(reset_query, {"user_id": user_id})
        connection.execute(query, [
            {
                "user_id": item["user_id"],
                "response_id": item.get("response_id"),
                "question_id": item["question_id"],
                "answer_id": item.get("answer_id"),
            }
            for item in valid_responses
        ])
def get_user_response_rate(user_id: int):
    """
    Calculate CLAN question response rate for a user.

    Returns None when no questions were sent.
    """

    query = text("""
        WITH assigned AS (
            SELECT
                user_id,
                COUNT(DISTINCT question_id) AS questions_sent
            FROM public.user_persona_question_assignment
            WHERE user_id = :user_id
            GROUP BY user_id
        ),
        answered AS (
            SELECT
                user_id,
                COUNT(DISTINCT question_id) AS questions_answered
            FROM public.user_persona_question_responces
            WHERE user_id = :user_id
              AND status = 1
            GROUP BY user_id
        )
        SELECT
            a.user_id,
            a.questions_sent,
            COALESCE(r.questions_answered, 0) AS questions_answered,
            ROUND(
                COALESCE(r.questions_answered, 0) * 100.0
                / NULLIF(a.questions_sent, 0),
                2
            ) AS response_percentage
        FROM assigned a
        LEFT JOIN answered r
            ON a.user_id = r.user_id;
    """)

    with engine.connect() as connection:
        row = connection.execute(
            query,
            {"user_id": user_id},
        ).fetchone()

    if not row:
        return None

    response_percentage = float(row.response_percentage)

    return {
        "user_id": row.user_id,
        "questions_sent": row.questions_sent,
        "questions_answered": row.questions_answered,
        "response_percentage": response_percentage,
        "notification_type": (
            "IMPROVEMENT"
            if response_percentage < 60
            else "POSITIVE"
        ),
    }


def get_responses(user_id: int, db_engine=engine) -> list[dict]:
    query = text("""
         SELECT r.question_id, q.question AS question_text, r.answer_id,
             a.answer_text
        FROM public.user_persona_question_responces r
        JOIN public.user_persona_question q ON q.id = r.question_id
        JOIN public.user_persona_question_answers a ON a.id = r.answer_id AND a.question_id = r.question_id
        WHERE r.status = 1 AND r.user_id = :user_id
        ORDER BY r.question_id, r.answer_id
    """)
    with db_engine.connect() as connection:
        return [
            {
                "question_id": row["question_id"],
                "question": row["question_text"],
                "answer_id": row["answer_id"],
                "answer": row["answer_text"],
            }
            for row in connection.execute(query, {"user_id": user_id}).mappings()
        ]
