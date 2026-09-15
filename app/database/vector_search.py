from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from app.database.connection import engine


EMBEDDING_DIMENSIONS = 384


@dataclass
class SearchResult:
    payload: dict[str, Any]
    score: float


def _vector_literal(vector: list[float]) -> str:
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS}-dimensional embedding, "
            f"got {len(vector)} dimensions"
        )

    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    return [str(item) for item in value if item is not None]


def _payload_from_mapping(row: dict[str, Any]) -> dict[str, Any]:
    description = str(row.get("description") or "")

    language = (
        str(row.get("language_code") or "en")
        .strip()
        .lower()
        or "en"
    )

    return {
        "video_id": int(row["video_id"]),
        "title": str(row.get("title") or ""),
        "creator_name": str(row.get("creator_name") or ""),
        "creator_role": str(row.get("creator_role") or ""),
        "creator_region": str(row.get("creator_region") or ""),
        "lead_indicators": _normalize_list(row.get("lead_indicators")),
        "summary": description,
        "key_lesson": "",
        "problem_solved": description,
        "sales_phase": "all",
        "experience_level": "all",
        "language": language,
        "language_id": row.get("language_id"),
        "language_name": language,
        "description": description,
        "thumbnail_url": str(row.get("thumbnail_url") or ""),
    }


def query_points(
    query_vector: list[float],
    limit: int = 5,
    lead_indicator: str | None = None,
    user_id: int | None = None,
) -> list[SearchResult]:

    vector_value = _vector_literal(query_vector)

    params = {
        "query_vector": vector_value,
        "limit": int(limit),
    }

    where_clause = ""

    # Optional KII filter
    if lead_indicator:
        where_clause += """
            AND EXISTS (
                SELECT 1
                FROM public.kii_content_relation r
                JOIN public.kii_master km
                    ON km.id = r.kii_id
                WHERE r.content_id = c.id
                  AND COALESCE(r.status, 1) = 1
                  AND km.status = 1
                  AND lower(replace(km.kii_name, ' ', '_'))
                      = :lead_indicator
            )
        """

        params["lead_indicator"] = (
            str(lead_indicator)
            .strip()
            .lower()
            .replace(" ", "_")
        )

    # Optional user-language filter
    if user_id is not None:
        where_clause += """
            AND c.language_id IN (
                SELECT ul.language_id
                FROM public.user_language ul
                WHERE ul.user_id = :user_id
            )
        """

        params["user_id"] = int(user_id)

    sql = text(
        f"""
        WITH latest_embeddings AS (
            SELECT DISTINCT ON (ce.content_id)
                ce.id,
                ce.content_id,
                ce.embedding
            FROM public.content_embeddings ce
            WHERE ce.embedding IS NOT NULL
              AND vector_dims(ce.embedding) = {EMBEDDING_DIMENSIONS}
            ORDER BY ce.content_id, ce.id DESC
        ),

        content_indicators AS (
            SELECT
                r.content_id,
                array_agg(
                    DISTINCT lower(replace(km.kii_name, ' ', '_'))
                ) AS lead_indicators
            FROM public.kii_content_relation r
            JOIN public.kii_master km
                ON km.id = r.kii_id
            WHERE COALESCE(r.status, 1) = 1
              AND km.status = 1
            GROUP BY r.content_id
        )

        SELECT
            c.id AS video_id,
            c.title,
            c.description,
            c.thumbnail_url,

            COALESCE(l.code, 'en') AS language_code,
            c.language_id,

            COALESCE(u.name, '') AS creator_name,
            COALESCE(u.zone, '') AS creator_region,
            '' AS creator_role,

            COALESCE(
                ci.lead_indicators,
                ARRAY[]::text[]
            ) AS lead_indicators,

            (
                latest_embeddings.embedding::vector({EMBEDDING_DIMENSIONS})
                <=>
                CAST(:query_vector AS vector({EMBEDDING_DIMENSIONS}))
            ) AS distance

        FROM latest_embeddings

        JOIN public.content c
            ON c.id = latest_embeddings.content_id

        LEFT JOIN public.language l
            ON l.id = c.language_id

        LEFT JOIN public.expert_user e
            ON e.user_id = c.created_by

        LEFT JOIN public."user" u
            ON u.id = e.user_id

        LEFT JOIN content_indicators ci
            ON ci.content_id = c.id

        WHERE c.status = 1

        {where_clause}

        ORDER BY distance

        LIMIT :limit
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()

    results = []

    for row in rows:
        distance = float(row["distance"])

        # Convert cosine distance to similarity score
        score = max(0.0, 1.0 - distance)

        results.append(
            SearchResult(
                payload=_payload_from_mapping(dict(row)),
                score=score,
            )
        )

    return results


def search_videos(
    kii_id: int,
    language_id: int | list[int] | None,
    query_embedding: list[float],
    db_engine=engine,
    user_id: int | None = None,
) -> dict[str, Any] | None:
    language_ids = (
        [int(language_id)]
        if isinstance(language_id, int)
        else [int(value) for value in (language_id or [])]
    )
    language_filter = ""
    params: dict[str, Any] = {
        "kii_id": kii_id,
        "user_id": user_id,
        "embedding": _vector_literal(query_embedding),
    }
    if language_ids:
        language_params = []
        for index, value in enumerate(language_ids):
            key = f"language_id_{index}"
            language_params.append(f":{key}")
            params[key] = value
        language_filter = f"AND c.language_id IN ({', '.join(language_params)})"
    else:
        language_filter = "AND 1 = 0"

    sql = text(f"""
        SELECT c.id AS video_id, c.title, c.description, c.created_by AS creator_name, c.language_id
        FROM public.kii_content_relation kr
        JOIN public.content c ON c.id = kr.content_id AND c.status = 1
        JOIN public.language l ON l.id = c.language_id
        JOIN public.content_embeddings ce ON ce.content_id = c.id
        WHERE kr.kii_id = :kii_id AND kr.status = 1
          {language_filter}
          AND (
              :user_id IS NULL
              OR NOT EXISTS (
                  SELECT 1
                  FROM public.user_watched_history_copy wh
                  WHERE wh.user_id = :user_id
                    AND wh.content_id = c.id
                    AND wh.duration_seconds = wh.played_seconds
              )
          )
        ORDER BY ce.embedding <=> CAST(:embedding AS vector), c.id
        LIMIT 1
    """)
    with db_engine.connect() as connection:
        row = connection.execute(sql, params).mappings().first()
    return dict(row) if row else None