from sqlalchemy import text

from app.recommendation import recommendation as rec
from app.database.connection import engine


class FakeEmbeddingModel:
    def __call__(self, text):
        return [0.0] * 384


class FakeEmbeddingResponse:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddingsClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def create(self, model, input):
        return type(
            "Response",
            (),
            {
                "data": [
                    FakeEmbeddingResponse([0.0] * 384)
                ]
            },
        )()


def test_embed_text_returns_384_dimensions(monkeypatch):
    """
    Verify that embed_text() returns a 384-dimensional
    embedding when Sentence Transformers is selected.
    """

    monkeypatch.setattr(
        rec,
        "_resolve_embedding_provider",
        lambda: "sentence_transformers",
    )

    monkeypatch.setattr(
        rec,
        "_embed_with_sentence_transformers",
        lambda text: [0.0] * 384,
    )

    result = rec.embed_text(
        "Customers Interested: improve performance"
    )

    assert len(result) == 384
    assert all(value == 0.0 for value in result)


def test_sentence_transformers_embedding_path_does_not_use_openai(
    monkeypatch,
):
    """
    Verify that the Sentence Transformer path is used
    and OpenAI embedding is not called.
    """

    class FakeEmbedding:
        def flatten(self):
            return self

        def tolist(self):
            return [0.0] * 384

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            assert model_name == "all-MiniLM-L6-v2"

        def encode(self, text):
            assert text == "local query"
            return FakeEmbedding()

    monkeypatch.setattr(
        rec,
        "EMBEDDING_PROVIDER",
        "sentence_transformers",
    )

    monkeypatch.setattr(
        rec,
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2",
    )

    monkeypatch.setattr(
        rec,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    monkeypatch.setattr(
        rec,
        "_embed_with_openai",
        lambda text: (
            (_ for _ in ()).throw(
                AssertionError("OpenAI was called")
            )
        ),
    )

    rec._get_sentence_transformer.cache_clear()

    result = rec.embed_text("local query")

    assert len(result) == 384
    assert all(value == 0.0 for value in result)


def test_user_language_id_maps_to_language_table():
    """
    Verify the relationship:

        user_language.language_id
                    ↓
              language.id

    and verify that the language information can be retrieved.
    """

    user_id = 953

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    ul.user_id,
                    ul.language_id,
                    l.id AS language_table_id,
                    l.code AS language_code
                FROM public.user_language ul
                JOIN public.language l
                    ON l.id = ul.language_id
                WHERE ul.user_id = :user_id
                """
            ),
            {"user_id": user_id},
        ).mappings().all()

    assert rows, (
        f"No language mapping found for user_id={user_id}"
    )

    for row in rows:
        assert row["user_id"] == user_id
        assert row["language_id"] == row["language_table_id"]
        assert row["language_code"] is not None

        print(
            f"User {user_id} -> "
            f"language_id={row['language_id']} -> "
            f"language_code={row['language_code']}"
        )


def test_recommend_for_user_uses_weakest_kii_and_video_language(
    monkeypatch,
):
    """
    Verify that recommend_for_user():

    1. Gets the weakest KII.
    2. Gets user's video language IDs.
    3. Generates a 384-dimensional embedding.
    4. Passes KII + language IDs to search_videos().
    5. Returns the selected video.
    """

    monkeypatch.setattr(
        rec,
        "calculate_performance",
        lambda user_id: {
            "improvement_area": {
                "kii_id": 121,
                "kii_name": "Customers Interested",
                "performance_percentage": 38.1,
            }
        },
    )

    monkeypatch.setattr(
        rec,
        "get_user",
        lambda user_id, db_engine=None: {
            "video_language_ids": [2, 5],
        },
    )

    monkeypatch.setattr(
        rec,
        "embed_text",
        lambda text: [0.1] * 384,
    )

    captured = {}

    def fake_search_videos(
        kii_id,
        language_id,
        query_embedding,
        db_engine,
        user_id,
    ):
        captured["kii_id"] = kii_id
        captured["language_id"] = language_id
        captured["embedding_len"] = len(query_embedding)
        captured["user_id"] = user_id

        return {
            "video_id": 123,
            "title": "Improve Customer Conversations",
            "description": (
                "Practical guidance for customer engagement."
            ),
            "language_id": 2,
        }

    monkeypatch.setattr(
        rec,
        "search_videos",
        fake_search_videos,
    )

    result = rec.recommend_for_user(953)

    assert result is not None

    assert result["kii_id"] == 121
    assert result["kii_name"] == "Customers Interested"
    assert result["performance_percentage"] == 38.1
    assert result["video_id"] == 123

    assert captured["kii_id"] == 121
    assert captured["language_id"] == [2, 5]
    assert captured["embedding_len"] == 384
    assert captured["user_id"] == 953


def test_recommend_for_user_returns_none_when_no_match(
    monkeypatch,
):
    """
    Verify that recommend_for_user() returns None
    when no suitable video is found.
    """

    monkeypatch.setattr(
        rec,
        "calculate_performance",
        lambda user_id: {
            "improvement_area": {
                "kii_id": 120,
                "kii_name": "Productivity",
                "performance_percentage": 55.0,
            }
        },
    )

    monkeypatch.setattr(
        rec,
        "get_user",
        lambda user_id, db_engine=None: {
            "video_language_ids": [2],
        },
    )

    monkeypatch.setattr(
        rec,
        "embed_text",
        lambda text: [0.0] * 384,
    )

    monkeypatch.setattr(
        rec,
        "search_videos",
        lambda **kwargs: None,
    )

    assert rec.recommend_for_user(953) is None