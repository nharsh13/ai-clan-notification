from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable

from app.database.connection import engine
from app.database.user_repository import get_user
from app.database.vector_search import search_videos
from app.performance.performance import calculate_performance
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
logger = logging.getLogger(__name__)


def _embed_with_sentence_transformers(text: str) -> list[float]:
    model = _get_sentence_transformer()
    embedding = model.encode(text)
    return [float(value) for value in embedding.flatten().tolist()]


@lru_cache(maxsize=1)
def _get_sentence_transformer() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_text(text: str) -> list[float]:
    embedding = _embed_with_sentence_transformers(text)

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding length mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(embedding)}"
        )

    return embedding


def recommend_video(
    performance: Any,
    language_id: int | list[int] | None,
    embed: Callable[[str], list[float]],
    db_engine,
    user_id: int | None = None,
) -> dict[str, Any] | None:
    if performance is None:
        return None

    query = f"How to improve {performance.kii_name}"
    try:
        query_embedding = embed(query)
    except Exception:
        logger.exception(
            "Video recommendation query embedding failed for KII %s",
            performance.kii_id,
        )
        raise

    if len(query_embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding length mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(query_embedding)}"
        )

    return search_videos(
        kii_id=performance.kii_id,
        language_id=language_id,
        query_embedding=query_embedding,
        db_engine=db_engine,
        user_id=user_id,
    )


def recommend_for_user(user_id: int, db_engine=engine) -> dict[str, Any] | None:
    result = calculate_performance(user_id)
    weakest = result["improvement_area"]

    user = get_user(user_id, db_engine)
    language_id = None if user is None else user.get("video_language_ids")

    query = (
        f"{weakest['kii_name']}: improve performance, "
        f"current performance {float(weakest['performance_percentage']):.2f}%"
    )
    query_embedding = embed_text(query)

    video = search_videos(
        kii_id=weakest["kii_id"],
        language_id=language_id,
        query_embedding=query_embedding,
        db_engine=db_engine,
        user_id=user_id,
    )

    if video is None:
        return None

    return {
        "kii_id": int(weakest["kii_id"]),
        "kii_name": weakest["kii_name"],
        "performance_percentage": float(weakest["performance_percentage"]),
        "video_id": int(video["video_id"]),
        "title": video.get("title"),
        "description": video.get("description"),
        "language_id": video.get("language_id"),
    }