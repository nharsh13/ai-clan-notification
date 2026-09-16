from app.database import vector_search


class FakeResult:
	def __init__(self, row):
		self.row = row

	def mappings(self):
		return self

	def first(self):
		return self.row


class FakeConnection:
	def __init__(self, history, candidates):
		self.history = history
		self.candidates = candidates
		self.query = None

	def execute(self, query, params):
		self.query = str(query)
		eligible = [
			candidate
			for candidate in self.candidates
			if not any(
				record["user_id"] == params["user_id"]
				and record["content_id"] == candidate["video_id"]
				and record["duration_seconds"] == record["played_seconds"]
				for record in self.history
			)
		]
		return FakeResult(eligible[0] if eligible else None)


class FakeEngine:
	def __init__(self, connection):
		self.connection = connection

	def connect(self):
		return self

	def __enter__(self):
		return self.connection

	def __exit__(self, exc_type, exc_value, traceback):
		return False


def run_search(history, candidates):
	connection = FakeConnection(history, candidates)
	result = vector_search.search_videos(
		kii_id=121,
		language_id=2,
		query_embedding=[0.0] * 384,
		db_engine=FakeEngine(connection),
		user_id=953,
	)
	return result


def test_search_videos_excludes_fully_watched_video():
	result = run_search(
		[{"user_id": 953, "content_id": 123, "duration_seconds": 120, "played_seconds": 120}],
		[{"video_id": 123}, {"video_id": 456}],
	)

	assert result["video_id"] == 456


def test_search_videos_requires_non_null_embedding():
	connection = FakeConnection([], [{"video_id": 123}])
	vector_search.search_videos(
		kii_id=121,
		language_id=2,
		query_embedding=[0.0] * 384,
		db_engine=FakeEngine(connection),
		user_id=953,
	)

	assert "ce.embedding IS NOT NULL" in connection.query


def test_search_videos_keeps_partially_watched_video_eligible():
	result = run_search(
		[{"user_id": 953, "content_id": 123, "duration_seconds": 120, "played_seconds": 80}],
		[{"video_id": 123}],
	)

	assert result["video_id"] == 123


def test_search_videos_keeps_never_watched_video_eligible():
	result = run_search([], [{"video_id": 123}])

	assert result["video_id"] == 123


def test_search_videos_returns_none_when_all_videos_are_fully_watched():
	result = run_search(
		[{"user_id": 953, "content_id": 123, "duration_seconds": 120, "played_seconds": 120}],
		[{"video_id": 123}],
	)

	assert result is None
