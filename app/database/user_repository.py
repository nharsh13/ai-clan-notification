from typing import Any

from sqlalchemy import text

from app.database.connection import engine


def get_user(user_id: int, db_engine=engine) -> dict[str, Any] | None:
    query = text(
        """
        SELECT u.id AS user_id,
               COALESCE(NULLIF(u.name, ''), '') AS user_name,
               e.account_id,
               u.app_language_id,
               app_lang.language_code AS app_language_code,
               ARRAY_AGG(DISTINCT ul.language_id) FILTER (WHERE ul.language_id IS NOT NULL)
                   AS video_language_ids,
               ARRAY_AGG(DISTINCT l.code) FILTER (WHERE l.code IS NOT NULL)
                   AS video_language_codes
        FROM public."user" AS u
        LEFT JOIN public.expert_user AS e ON e.user_id = u.id
        LEFT JOIN public.md_app_languages AS app_lang
            ON app_lang.id = u.app_language_id
        LEFT JOIN public.user_language AS ul ON ul.user_id = u.id
        LEFT JOIN public.language AS l ON l.id = ul.language_id
        WHERE u.id = :user_id
        GROUP BY u.id, u.name, e.account_id, u.app_language_id, app_lang.language_code
        """
    )
    with db_engine.connect() as connection:
        row = connection.execute(query, {"user_id": user_id}).mappings().first()
    return dict(row) if row else None
from typing import Any

from sqlalchemy import text

from app.database.connection import engine


def get_user_type_id(user_id: int) -> int:
    query = text(
        """
        SELECT user_type_id
        FROM public."user"
        WHERE id = :user_id
        """
    )

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"user_id": user_id},
        ).scalar_one_or_none()

    if result is None:
        raise ValueError(
            f"User not found or user_type_id is missing: {user_id}"
        )

    return int(result)


def get_user_language(user_id: int) -> str:
    query = text(
        """
        SELECT COALESCE(app_lang.language_code, 'hi')
        FROM public."user" u
        LEFT JOIN public.md_app_languages app_lang
            ON app_lang.id = u.app_language_id
        WHERE u.id = :user_id
        """
    )

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"user_id": user_id},
        ).scalar_one_or_none()

    if result is None:
        raise ValueError(
            f"User not found or language is missing: {user_id}"
        )

    return str(result).strip().lower()


def get_last_7_days_kii_performance(
    user_id: int,
) -> list[dict[str, Any]]:
    query = text(
        """
        WITH latest_date AS (
            SELECT MAX(ki.kii_date) AS max_date
            FROM public.key_input_indicators ki
            WHERE ki.user_id = :user_id
              AND ki.account_id = 14
              AND ki.status = 1
              AND ki.kii_id IN (117, 118, 119, 120, 121)
        ),

        user_targets AS (
            SELECT DISTINCT ON (kit.kii_id)
                kit.kii_id,
                kit.kii_target AS monthly_target
            FROM public.key_input_target kit
            WHERE kit.user_id = :user_id
              AND kit.account_id = 14
              AND kit.status = 1
            ORDER BY
                kit.kii_id,
                kit.target_date DESC,
                kit.created_at DESC
        ),

        actual_performance AS (
            SELECT
                ki.kii_id,
                SUM(COALESCE(ki.kill_value, 0)) AS seven_day_actual
            FROM public.key_input_indicators ki

            CROSS JOIN latest_date ld

            WHERE ki.user_id = :user_id
              AND ki.account_id = 14
              AND ki.status = 1
              AND ki.kii_id IN (117, 118, 119, 120, 121)

              AND ki.kii_date BETWEEN
                    ld.max_date - INTERVAL '6 days'
                    AND ld.max_date

            GROUP BY ki.kii_id
        )

        SELECT
            km.id AS kii_id,
            km.kii_name,

            ut.monthly_target,

            COALESCE(
                ut.monthly_target / 22.0,
                km.daily_target
            ) AS daily_target,

            COALESCE(
                ap.seven_day_actual,
                0
            ) AS seven_day_actual

        FROM public.kii_master km

        LEFT JOIN user_targets ut
            ON ut.kii_id = km.id

        LEFT JOIN actual_performance ap
            ON ap.kii_id = km.id

        WHERE km.id IN (117, 118, 119, 120, 121)
          AND km.account_id = 14
          AND km.status = 1

        ORDER BY km.id;
        """
    )

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"user_id": user_id},
        ).mappings().all()

    return [dict(row) for row in result]


def get_user_video_language_ids(
    user_id: int,
) -> list[int]:
    query = text(
        """
        SELECT language_id
        FROM public.user_language
        WHERE user_id = :user_id
        """
    )

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"user_id": user_id},
        ).scalars().all()

    return [int(language_id) for language_id in result]


def get_kii_videos(
    kii_id: int,
    user_id: int,
) -> list[dict[str, Any]]:
    query = text(
        """
        SELECT DISTINCT
            c.id AS video_id,
            c.title,
            c.description,
            c.language_id

        FROM public.kii_content_relation r

        JOIN public.content c
            ON c.id = r.content_id

        WHERE r.kii_id = :kii_id
          AND r.status = 1
          AND c.status = 1
          AND c.language_id IN (
              SELECT ul.language_id
              FROM public.user_language ul
              WHERE ul.user_id = :user_id
          )

        ORDER BY c.id;
        """
    )

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {
                "kii_id": kii_id,
                "user_id": user_id,
            },
        ).mappings().all()

    return [dict(row) for row in result]