from __future__ import annotations

import os
from typing import Any, Callable

from app.database.connection import engine
from app.database.user_repository import get_user
from app.database.vector_search import search_videos
from app.performance.performance import calculate_performance
from sentence_transformers import SentenceTransformer
EMBEDDING_DIMENSIONS = 384


def _resolve_embedding_provider() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    try:
        import sentence_transformers  # type: ignore
        _ = sentence_transformers
        return "sentence_transformers"
    except Exception:
        return "fallback"


def _embed_with_openai(text: str) -> list[float]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        input=text,
    )
    embedding = response.data[0].embedding
    return [float(value) for value in embedding]


def _embed_with_sentence_transformers(text: str) -> list[float]:
    model_name = os.getenv("EMBED_MODEL", os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    model = SentenceTransformer(model_name)
    embedding = model.encode(text)
    return [float(value) for value in embedding.flatten().tolist()]


def embed_text(text: str) -> list[float]:
    provider = _resolve_embedding_provider()

    if provider == "openai":
        embedding = _embed_with_openai(text)
    elif provider == "sentence_transformers":
        embedding = _embed_with_sentence_transformers(text)
    else:
        embedding = [0.0] * EMBEDDING_DIMENSIONS

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding length mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(embedding)}"
        )

    return embedding


def recommend_video(
    performance: Any,
    language_id: int | None,
    embed: Callable[[str], list[float]],
    db_engine,
    user_id: int | None = None,
) -> dict[str, Any] | None:
    if performance is None:
        return None

    query = f"{performance.kii_name}: improve performance"
    query_embedding = embed(query)

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
    language_id = None if user is None else user.get("video_language_id")

    query = f"{weakest['kii_name']}: improve performance"
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