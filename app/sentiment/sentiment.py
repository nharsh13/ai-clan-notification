from sqlalchemy import text

from app.database.connection import engine


def get_user_answered_questions(user_id: int):
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

    with engine.connect() as connection:
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


def prepare_user_qa(user_id: int):
    """
    Group all answered Q&A for a user by question.
    """

    rows = get_user_answered_questions(user_id)

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


def get_next_sentiment_response(user_id: int, db_engine=engine) -> dict | None:
    query = text("""
        SELECT
            r.id AS response_id,
            r.user_id,
            r.question_id,
            q.question AS question,
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
          AND NOT EXISTS (
              SELECT 1
              FROM public.sentiment_notification_history h
              WHERE h.user_id = r.user_id
                AND h.response_id = r.id
                AND h.status = 1
          )
          AND NOT EXISTS (
              SELECT 1
              FROM public.sentiment_notification_history h
              WHERE h.user_id = :user_id
                AND h.status = 1
                AND h.created_at::date = CURRENT_DATE
          )
        ORDER BY r.created_at, r.id
        LIMIT 1
    """)

    with db_engine.connect() as connection:
        row = connection.execute(query, {"user_id": user_id}).mappings().first()

    return dict(row) if row else None


def save_sentiment_notification_history(
    response: dict,
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

    with db_engine.begin() as connection:
        connection.execute(query, {
            "user_id": response["user_id"],
            "response_id": response["response_id"],
            "question_id": response["question_id"],
            "answer_id": response["answer_id"],
        })


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