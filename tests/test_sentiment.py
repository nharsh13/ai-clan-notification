from app.sentiment.sentiment import (
    get_user_answered_questions,
    prepare_user_qa,
    get_user_response_rate,
)

from app.llm.notification_generator import (
    build_notification_1_prompt,
    build_notification_2_prompt,
    validate_notification_1,
    validate_notification_2,
)


USER_ID = 953


# ============================================================
# TEST 1
# GET USER ANSWERED QUESTIONS
# ============================================================

def test_get_user_answered_questions():

    results = get_user_answered_questions(USER_ID)

    print("\n" + "=" * 70)
    print("TEST 1 - ANSWERED Q&A")
    print("=" * 70)

    for item in results:
        print(item)

    assert isinstance(results, list)

    for item in results:
        assert item["user_id"] == USER_ID
        assert item["question_id"] is not None
        assert item["question"] is not None
        assert item["answer_id"] is not None
        assert item["answer"] is not None


# ============================================================
# TEST 2
# PREPARE / GROUP USER Q&A
# ============================================================

def test_prepare_user_qa():

    result = prepare_user_qa(USER_ID)

    print("\n" + "=" * 70)
    print("TEST 2 - PREPARED Q&A")
    print("=" * 70)

    print(result)

    assert result["user_id"] == USER_ID
    assert isinstance(result["questions"], list)

    for question in result["questions"]:

        assert "question_id" in question
        assert "responses" in question
        assert len(question["responses"]) > 0

        for response in question["responses"]:

            assert "question" in response
            assert "answer_id" in response
            assert "answer" in response


# ============================================================
# TEST 3
# NOTIFICATION 2 PROMPT
# ============================================================

def test_notification_2_prompt():

    prepared_qa = prepare_user_qa(USER_ID)

    prompt = build_notification_2_prompt(
        user_name="Test User",
        language="English",
        prepared_qa=prepared_qa,
    )

    print("\n" + "=" * 70)
    print("TEST 3 - NOTIFICATION 2 PROMPT")
    print("=" * 70)

    print(prompt)

    assert prompt
    assert "Test User" in prompt
    assert "English" in prompt
    assert "ALL Q&A" in prompt
    assert "ONE notification" in prompt


# ============================================================
# TEST 4
# NOTIFICATION 1 RESPONSE RATE
# ============================================================

def test_user_response_rate():

    result = get_user_response_rate(USER_ID)

    print("\n" + "=" * 70)
    print("TEST 4 - NOTIFICATION 1 RESPONSE RATE")
    print("=" * 70)

    print(result)

    if result is None:

        print("No questions sent for this user.")

        return

    assert result["user_id"] == USER_ID

    assert result["questions_sent"] >= 0

    assert result["questions_answered"] >= 0

    assert 0 <= result["response_percentage"] <= 100

    if result["response_percentage"] < 60:

        assert result["notification_type"] == "IMPROVEMENT"

    else:

        assert result["notification_type"] == "POSITIVE"


# ============================================================
# TEST 5
# NOTIFICATION 1 PROMPT
# ============================================================

def test_notification_1_prompt():

    response_data = get_user_response_rate(USER_ID)

    print("\n" + "=" * 70)
    print("TEST 5 - NOTIFICATION 1 PROMPT")
    print("=" * 70)

    if response_data is None:

        print("No questions sent for this user.")

        return

    print("\nNotification 1 Data:")
    print(response_data)

    prompt = build_notification_1_prompt(
        user_name="Test User",
        language="English",
        response_data=response_data,
    )

    print("\nNotification 1 Prompt:")
    print(prompt)

    assert prompt

    assert "Test User" in prompt

    assert "English" in prompt

    assert str(
        response_data["questions_sent"]
    ) in prompt

    assert str(
        response_data["questions_answered"]
    ) in prompt

    assert str(
        response_data["response_percentage"]
    ) in prompt

    assert response_data["notification_type"] in prompt


# ============================================================
# TEST 6
# NOTIFICATION 1 VALIDATION
# ============================================================

def test_notification_1_validation():

    valid_notification = {
        "title": "Test User, Keep Engaging",
        "description": "Continue participating in CLAN.",
    }

    validate_notification_1(valid_notification)

    print("\n" + "=" * 70)
    print("TEST 6 - NOTIFICATION 1 VALIDATION")
    print("=" * 70)

    print("Notification 1 validation passed.")


# ============================================================
# TEST 7
# NOTIFICATION 2 VALIDATION
# ============================================================

def test_notification_2_validation():

    valid_notification = {
        "title": "Test User, Build Stronger Results",
        "description": "Keep developing your workplace skills.",
    }

    validate_notification_2(valid_notification)

    print("\n" + "=" * 70)
    print("TEST 7 - NOTIFICATION 2 VALIDATION")
    print("=" * 70)

    print("Notification 2 validation passed.")