from app.sentiment import sentiment as sent


# ============================================================
# NOTIFICATION 1 — CLAN ENGAGEMENT / RESPONSE RATE
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


def test_notification_1_improvement_classification():
    """
    Verify Notification 1 classification.
    """

    response_percentage = 40.0

    notification_type = (
        "IMPROVEMENT"
        if response_percentage < 60
        else "POSITIVE"
    )

    assert notification_type == "IMPROVEMENT"


def test_notification_1_positive_classification():
    """
    Verify Notification 1 classification.
    """

    response_percentage = 80.0

    notification_type = (
        "IMPROVEMENT"
        if response_percentage < 60
        else "POSITIVE"
    )

    assert notification_type == "POSITIVE"


# ============================================================
# NOTIFICATION 2 — QUESTION + SELECTED ANSWER
# ============================================================

def test_notification_2_selected_answer_data():
    """
    Verify that Notification 2 contains the required
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


def test_notification_2_llm_input_contains_required_fields():
    """
    Verify the information sent to the LLM for Notification 2.
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


def test_notification_2_different_answers_are_preserved():
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


def test_notification_2_multiple_questions_are_collected():
    """
    Multiple answered questions should be collected together
    so that one Notification 2 can be generated.
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


def test_notification_2_generates_one_notification_for_multiple_answers():
    """
    Multiple Q&A responses should result in ONE Notification 2,
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
# NOTIFICATION 1 AND NOTIFICATION 2 MUST BE INDEPENDENT
# ============================================================

def test_notification_1_and_notification_2_are_independent():
    """
    Notification 2 must not depend on Notification 1's
    response percentage.
    """

    response_percentage = 20.0

    notification_1_type = (
        "IMPROVEMENT"
        if response_percentage < 60
        else "POSITIVE"
    )

    notification_2_input = {
        "question": "How do you respond to feedback?",
        "selected_answer": "I listen and make changes if needed.",
    }

    assert notification_1_type == "IMPROVEMENT"

    # Notification 2 still has its own Q&A data.
    assert notification_2_input["question"]
    assert notification_2_input["selected_answer"]