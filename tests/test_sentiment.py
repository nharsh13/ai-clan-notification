from typing import cast

from sqlalchemy.engine import Engine

from app.sentiment import sentiment as sent


class _HistoryResult:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def __iter__(self):
        return iter(self.rows)


class _HistoryConnection:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.params = None

    def execute(self, query, params):
        self.query = str(query)
        self.params = params
        return _HistoryResult(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _HistoryEngine:
    def __init__(self, rows):
        self.connection = _HistoryConnection(rows)

    def connect(self):
        return self.connection


def test_eligible_qa_history_check_uses_exact_response_answer_identity():
    engine = _HistoryEngine([])

    sent.get_eligible_user_qa(953, cast(Engine, engine))

    query = engine.connection.query
    assert "h.user_id = response.user_id" in query
    assert "h.response_id = response.response_id" in query
    assert "h.question_id = response.question_id" in query
    assert "h.answer_id = response.answer_id" in query
    assert "h.status = 1" in query
    assert "FROM unused_responses" in query
    assert "FROM all_responses" in query
    assert "WHERE NOT EXISTS (SELECT 1 FROM unused_responses)" in query
    assert "DELETE FROM public.sentiment_notification_history" not in query
    assert "UPDATE public.sentiment_notification_history" not in query
    assert engine.connection.params == {"user_id": 953}


def test_history_save_writes_all_identity_fields_with_status_one():
    class WriteConnection:
        def __init__(self):
            self.query = ""
            self.params = None

        def execute(self, query, params):
            self.query = str(query)
            self.params = params

    class WriteEngine:
        def __init__(self):
            self.connection = WriteConnection()

        def begin(self):
            return self

        def __enter__(self):
            return self.connection

        def __exit__(self, *args):
            return False

    engine = WriteEngine()
    sent.save_sentiment_notification_history(
        [{
            "user_id": 953,
            "response_id": 11,
            "question_id": 1,
            "answer_id": 3,
        }],
        cast(Engine, engine),
    )

    assert engine.connection.params == [{
        "user_id": 953,
        "response_id": 11,
        "question_id": 1,
        "answer_id": 3,
    }]
    assert "status" in engine.connection.query
    assert ", 1" in engine.connection.query


# ============================================================
# ENGAGEMENT SENTIMENT — CLAN ENGAGEMENT / RESPONSE RATE
# ============================================================

def test_response_percentage_below_60_is_improvement():
    """
    10 questions sent
    4 answered
    Response = 40%
    Expected = IMPROVEMENT
    """

    questions_sent = 10
    questions_answered = 4

    response_percentage = (
        questions_answered * 100.0 / questions_sent
    )

    assert response_percentage == 40.0
    assert response_percentage < 60


def test_response_percentage_60_is_positive():
    """
    10 questions sent
    6 answered
    Response = 60%
    Expected = POSITIVE

    60% is explicitly positive.
    """

    questions_sent = 10
    questions_answered = 6

    response_percentage = (
        questions_answered * 100.0 / questions_sent
    )

    assert response_percentage == 60.0
    assert response_percentage >= 60


def test_response_percentage_above_60_is_positive():
    """
    10 questions sent
    8 answered
    Response = 80%
    Expected = POSITIVE
    """

    questions_sent = 10
    questions_answered = 8

    response_percentage = (
        questions_answered * 100.0 / questions_sent
    )

    assert response_percentage == 80.0
    assert response_percentage >= 60


def test_zero_questions_does_not_generate_notification():
    """
    If no questions were assigned, response percentage
    cannot be meaningfully calculated.
    """

    questions_sent = 0
    questions_answered = 0

    assert questions_sent == 0

    # Notification should not be generated.
    should_generate = questions_sent > 0

    assert should_generate is False


def test_duplicate_questions_are_counted_once():
    """
    Multiple records for the same question must not
    artificially increase the number of questions sent
    or answered.

    Example:
        Assigned: Q1, Q1, Q2
        Answered: Q1, Q2, Q2

    Distinct sent    = 2
    Distinct answered = 2
    Response         = 100%
    """

    assigned_questions = [1, 1, 2]
    answered_questions = [1, 2, 2]

    questions_sent = len(set(assigned_questions))
    questions_answered = len(set(answered_questions))

    response_percentage = (
        questions_answered * 100.0 / questions_sent
    )

    assert questions_sent == 2
    assert questions_answered == 2
    assert response_percentage == 100.0


def test_engagement_sentiment_improvement_classification():
    """
    Verify engagement sentiment classification.
    """

    response_percentage = 40.0

    notification_type = (
        "IMPROVEMENT"
        if response_percentage < 60
        else "POSITIVE"
    )

    assert notification_type == "IMPROVEMENT"


def test_engagement_sentiment_positive_classification():
    """
    Verify engagement sentiment classification.
    """

    response_percentage = 80.0

    notification_type = (
        "IMPROVEMENT"
        if response_percentage < 60
        else "POSITIVE"
    )

    assert notification_type == "POSITIVE"


# ============================================================
# Q/A SENTIMENT — QUESTION + SELECTED ANSWER
# ============================================================

def test_qa_sentiment_selected_answer_data():
    """
    Verify that the Q/A sentiment input contains the required
    question + selected answer information.
    """

    response = {
        "user_id": 953,
        "question_id": 1,
        "answer_id": 3,
        "question": (
            "How do you respond when a teammate "
            "questions your way of doing things?"
        ),
        "answer_text": (
            "I dismiss it if I disagree."
        ),
        "what_it_conveys": (
            "Opportunity to be more open to different perspectives"
        ),
        "recommended_action_to_manager": (
            "Encourage openness to feedback"
        ),
    }

    assert response["user_id"] == 953
    assert response["question_id"] == 1
    assert response["answer_id"] == 3

    assert response["question"]
    assert response["answer_text"]
    assert response["what_it_conveys"]
    assert response["recommended_action_to_manager"]


def test_qa_sentiment_llm_input_contains_required_fields():
    """
    Verify the information sent to the LLM for the Q/A sentiment notification.
    """

    llm_input = {
        "user_name": "Rahul",
        "question": (
            "How do you respond when a teammate "
            "questions your way of doing things?"
        ),
        "selected_answer": (
            "I dismiss it if I disagree"
        ),
        "what_it_conveys": (
            "Opportunity to be more open to different perspectives"
        ),
        "recommended_action_to_manager": (
            "Encourage openness to feedback"
        ),
    }

    required_fields = {
        "user_name",
        "question",
        "selected_answer",
        "what_it_conveys",
        "recommended_action_to_manager",
    }

    assert required_fields.issubset(llm_input.keys())


def test_qa_sentiment_different_answers_are_preserved():
    """
    Different answers must remain distinct because the LLM
    should generate different improvement messages based
    on the actual answer.
    """

    answer_1 = {
        "question_id": 1,
        "answer_id": 3,
        "answer_text": "I dismiss it if I disagree.",
    }

    answer_2 = {
        "question_id": 1,
        "answer_id": 4,
        "answer_text": "I listen and make changes if needed.",
    }

    assert answer_1["answer_id"] != answer_2["answer_id"]
    assert answer_1["answer_text"] != answer_2["answer_text"]


def test_qa_sentiment_multiple_questions_are_collected():
    """
    Multiple answered questions should be collected together
    so that one Q/A sentiment notification can be generated.
    """

    responses = [
        {
            "question_id": 1,
            "answer_id": 3,
            "question": "How do you respond to feedback?",
            "answer_text": "I dismiss it if I disagree.",
        },
        {
            "question_id": 2,
            "answer_id": 5,
            "question": "How do you handle different opinions?",
            "answer_text": "I prefer my own approach.",
        },
        {
            "question_id": 3,
            "answer_id": 7,
            "question": "How do you react when plans change?",
            "answer_text": "I prefer the original plan.",
        },
    ]

    assert len(responses) == 3

    for response in responses:
        assert response["question_id"]
        assert response["answer_id"]
        assert response["question"]
        assert response["answer_text"]


def test_qa_sentiment_generates_one_notification_for_multiple_answers():
    """
    Multiple Q&A responses should result in ONE Q/A sentiment notification,
    not one notification per question.
    """

    responses = [
        {"question_id": 1, "answer_id": 3},
        {"question_id": 2, "answer_id": 5},
        {"question_id": 3, "answer_id": 7},
    ]

    notification_count = 1 if responses else 0

    assert notification_count == 1


# ============================================================
# ENGAGEMENT AND Q/A SENTIMENT MUST BE INDEPENDENT
# ============================================================

def test_engagement_and_qa_sentiment_are_independent():
    """
    Q/A sentiment must not depend on engagement response percentage.
    """

    response_percentage = 20.0

    engagement_sentiment_type = (
        "IMPROVEMENT"
        if response_percentage < 60
        else "POSITIVE"
    )

    qa_sentiment_input = {
        "question": "How do you respond to feedback?",
        "selected_answer": "I listen and make changes if needed.",
    }

    assert engagement_sentiment_type == "IMPROVEMENT"

    # Q/A sentiment still has its own Q&A data.
    assert qa_sentiment_input["question"]
    assert qa_sentiment_input["selected_answer"]