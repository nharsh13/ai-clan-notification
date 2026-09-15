from app.database import user_repository


class FakeResult:
    def mappings(self):
        return self

    def first(self):
        return {
            "user_id": 953,
            "user_name": "Ava Example",
            "account_id": 14,
            "app_language_id": 3,
            "app_language_code": "hi",
            "video_language_ids": [1, 2],
            "video_language_codes": ["en", "fr"],
        }


class FakeConnection:
    def __init__(self):
        self.query = None
        self.params = None

    def execute(self, query, params):
        self.query = str(query)
        self.params = params
        assert "u.user_name" not in self.query
        assert "u.name" in self.query
        assert "u.app_language_id" in self.query
        assert "md_app_languages" in self.query
        assert "user_language" in self.query
        return FakeResult()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeEngine:
    def __init__(self):
        self.connection = FakeConnection()

    def connect(self):
        return self.connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_get_user_returns_separate_app_and_video_languages():
    db_engine = FakeEngine()

    result = user_repository.get_user(953, db_engine)

    assert result["user_id"] == 953
    assert result["user_name"] == "Ava Example"
    assert result["app_language_code"] == "hi"
    assert result["video_language_ids"] == [1, 2]
    assert db_engine.connection.params == {"user_id": 953}