from sqlalchemy import text

from app.database.connection import engine


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
    """Return unused active Q&A, restarting from all active Q&A when exhausted."""

    query = text("""
        WITH all_responses AS (
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
              AND r.status = 1
        ), unused_responses AS (
            SELECT response.*
            FROM all_responses response
            WHERE NOT EXISTS (
                SELECT 1
                FROM public.sentiment_notification_history h
                WHERE h.user_id = response.user_id
                  AND h.response_id = response.response_id
                  AND h.question_id = response.question_id
                  AND h.answer_id = response.answer_id
                  AND h.status = 1
            )
        )
        SELECT response_id, user_id, question_id, question, answer_id, answer, selection_type
        FROM (
            SELECT response_id, user_id, question_id, question, answer_id, answer, created_at, 'NEW' AS selection_type
            FROM unused_responses
            UNION ALL
            SELECT response_id, user_id, question_id, question, answer_id, answer, created_at, 'REUSED' AS selection_type
            FROM all_responses
            WHERE NOT EXISTS (SELECT 1 FROM unused_responses)
        ) selected_responses
        ORDER BY created_at, response_id
    """)

    with db_engine.connect() as connection:
        return [dict(row) for row in connection.execute(query, {"user_id": user_id}).mappings()]


def get_next_sentiment_response(user_id: int, db_engine=engine) -> dict | None:
    query = text("""
        WITH selected_question AS (
            SELECT
                r.question_id,
                MIN(r.created_at) AS first_created_at,
                MIN(r.id) AS first_response_id
            FROM public.user_persona_question_responces r
            WHERE r.user_id = :user_id
              AND r.status = 1
              AND NOT EXISTS (
                  SELECT 1
                  FROM public.sentiment_notification_history h
                  WHERE h.user_id = r.user_id
                    AND h.question_id = r.question_id
                    AND h.status = 1
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM public.sentiment_notification_history h
                  WHERE h.user_id = :user_id
                    AND h.status = 1
                    AND h.created_at::date = CURRENT_DATE
              )
            GROUP BY r.question_id
            ORDER BY first_created_at, first_response_id
            LIMIT 1
        )
        SELECT
            r.id AS response_id,
            r.user_id,
            r.question_id,
            q.question AS question,
            r.answer_id,
            a.answer_text AS answer,
            r.created_at
        FROM public.user_persona_question_responces r
        JOIN selected_question selected
            ON selected.question_id = r.question_id
        JOIN public.user_persona_question q
            ON q.id = r.question_id
        JOIN public.user_persona_question_answers a
            ON a.id = r.answer_id
           AND a.question_id = r.question_id
        WHERE r.user_id = :user_id
          AND r.status = 1
        ORDER BY r.created_at, r.id
    """)

    with db_engine.connect() as connection:
        rows = connection.execute(query, {"user_id": user_id}).mappings().all()

    if not rows:
        return None

    responses = [dict(row) for row in rows]
    selected = responses[0]
    selected["responses"] = responses
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
    with db_engine.begin() as connection:
        connection.execute(query, [
            {
                "user_id": item["user_id"],
                "response_id": item["response_id"],
                "question_id": item["question_id"],
                "answer_id": item["answer_id"],
            }
            for item in responses
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