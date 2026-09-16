import json

import pytest

from app.llm import notification_generator as llm


# ============================================================
# FAKE OPENAI RESPONSE
# ============================================================

class FakeResponse:
    def __init__(self, content: str):
        self.output_text: str = content


class FakeResponses:
    def __init__(self, content: str):
        self.content: str = content
        self.last_model: str | None = None
        self.last_input: str | None = None

    def create(self, model: str, input: str) -> FakeResponse:
        self.last_model = model
        self.last_input = input

        return FakeResponse(self.content)


class FakeOpenAIClient:
    def __init__(self, content: str):
        self.responses = FakeResponses(content)


# ============================================================
# NOTIFICATION 1 — IMPROVEMENT
# ============================================================

def test_generate_notification_1_improvement(monkeypatch):
    """
    Response percentage < 60%
    Expected notification type: IMPROVEMENT
    """

    fake_output = {
        "title": "Rahul, Improve Your CLAN Engagement",
        "description": (
            "Answer more CLAN questions to reflect on your work "
            "and strengthen your performance."
        ),
    }

    fake_client = FakeOpenAIClient(
        json.dumps(fake_output)
    )

    monkeypatch.setattr(
        llm,
        "_get_openai_client",
        lambda: fake_client,
    )

    response_data = {
        "questions_sent": 10,
        "questions_answered": 4,
        "response_percentage": 40,
        "notification_type": "IMPROVEMENT",
    }

    result = llm.generate_notification_1(
        user_name="Rahul",
        language="English",
        response_data=response_data,
    )

    # Function must return a notification
    assert result is not None

    # Required fields
    assert "title" in result
    assert "description" in result

    assert isinstance(result["title"], str)
    assert isinstance(result["description"], str)

    assert result["title"].strip()
    assert result["description"].strip()

    # User name must be in title
    assert "Rahul" in result["title"]

    # Verify backend data reached LLM prompt
    assert fake_client.responses.last_input is not None

    prompt = fake_client.responses.last_input

    assert "40" in prompt
    assert "IMPROVEMENT" in prompt
    assert "10" in prompt
    assert "4" in prompt


# ============================================================
# NOTIFICATION 1 — POSITIVE
# ============================================================

def test_generate_notification_1_positive(monkeypatch):
    """
    Response percentage >= 60%
    Expected notification type: POSITIVE
    """

    fake_output = {
        "title": "Amit, Keep Going Strong!",
        "description": (
            "Great job staying engaged with CLAN. "
            "Keep reflecting and using these insights."
        ),
    }

    fake_client = FakeOpenAIClient(
        json.dumps(fake_output)
    )

    monkeypatch.setattr(
        llm,
        "_get_openai_client",
        lambda: fake_client,
    )

    response_data = {
        "questions_sent": 10,
        "questions_answered": 8,
        "response_percentage": 80,
        "notification_type": "POSITIVE",
    }

    result = llm.generate_notification_1(
        user_name="Amit",
        language="English",
        response_data=response_data,
    )

    assert result is not None

    assert "title" in result
    assert "description" in result

    assert isinstance(result["title"], str)
    assert isinstance(result["description"], str)

    assert result["title"].strip()
    assert result["description"].strip()

    assert "Amit" in result["title"]

    # Verify backend data reached LLM prompt
    assert fake_client.responses.last_input is not None

    prompt = fake_client.responses.last_input

    assert "80" in prompt
    assert "POSITIVE" in prompt
    assert "10" in prompt
    assert "8" in prompt


# ============================================================
# NOTIFICATION 1 — NO DATA
# ============================================================

def test_generate_notification_1_returns_none_without_data():
    """
    If response_data is empty, Notification 1 should not
    call the LLM and should return None.
    """

    result = llm.generate_notification_1(
        user_name="Rahul",
        language="English",
        response_data={},
    )

    assert result is None


# ============================================================
# NOTIFICATION 2
# ============================================================

def test_generate_notification_2(monkeypatch):
    """
    Verify Notification 2:

    Question + selected answer + context
                    ↓
                   LLM
                    ↓
            title + description
    """

    fake_output = {
        "title": "Rahul, Turn Feedback Into Better Results",
        "description": (
            "Stay open to different perspectives to strengthen "
            "collaboration and improve your approach."
        ),
    }

    fake_client = FakeOpenAIClient(
        json.dumps(fake_output)
    )

    monkeypatch.setattr(
        llm,
        "_get_openai_client",
        lambda: fake_client,
    )

    prepared_qa = {
        "responses": [
            {
                "question": "How do you respond to feedback?",
                "selected_answer": (
                    "I dismiss it if I disagree."
                ),
            },
            {
                "question": "How do you handle different opinions?",
                "selected_answer": (
                    "I prefer my own approach."
                ),
            },
        ]
    }

    result = llm.generate_notification_2(
        user_name="Rahul",
        language="English",
        prepared_qa=prepared_qa,
    )

    assert result is not None

    assert "title" in result
    assert "description" in result

    assert isinstance(result["title"], str)
    assert isinstance(result["description"], str)

    assert result["title"].strip()
    assert result["description"].strip()

    # User name must be in title
    assert "Rahul" in result["title"]

    # Verify Q&A reached the LLM prompt
    assert fake_client.responses.last_input is not None

    prompt = fake_client.responses.last_input

    assert "How do you respond to feedback?" in prompt
    assert "I dismiss it if I disagree." in prompt

    assert "How do you handle different opinions?" in prompt
    assert "I prefer my own approach." in prompt


# ============================================================
# NOTIFICATION 1 PROMPT
# ============================================================

def test_notification_1_prompt_contains_required_data():
    """
    Verify Notification 1 prompt contains the backend-calculated
    engagement data.
    """

    response_data = {
        "questions_sent": 10,
        "questions_answered": 4,
        "response_percentage": 40,
        "notification_type": "IMPROVEMENT",
    }

    prompt = llm.build_notification_1_prompt(
        user_name="Rahul",
        language="English",
        response_data=response_data,
    )

    assert isinstance(prompt, str)

    assert "Rahul" in prompt
    assert "English" in prompt

    assert "10" in prompt
    assert "4" in prompt
    assert "40" in prompt
    assert "IMPROVEMENT" in prompt

    # Notification 1 should not depend on individual Q&A
    assert "selected_answer" not in prompt


# ============================================================
# NOTIFICATION 2 PROMPT
# ============================================================

def test_notification_2_prompt_contains_user_and_qa():
    """
    Verify Notification 2 prompt contains user, language,
    and prepared Q&A data.
    """

    prepared_qa = {
        "responses": [
            {
                "question": "How do you respond to feedback?",
                "selected_answer": (
                    "I listen and make changes."
                ),
            }
        ]
    }

    prompt = llm.build_notification_2_prompt(
        user_name="Rahul",
        language="English",
        prepared_qa=prepared_qa,
    )

    assert isinstance(prompt, str)

    assert "Rahul" in prompt
    assert "English" in prompt

    assert "How do you respond to feedback?" in prompt
    assert "I listen and make changes." in prompt


# ============================================================
# VALIDATE NOTIFICATION 1
# ============================================================

def test_validate_notification_1_accepts_valid_output():
    """
    Valid Notification 1 output should pass validation.
    """

    notification = {
        "title": "Rahul, Keep Going",
        "description": "Continue engaging with CLAN.",
    }

    result = llm.validate_notification_1(notification)

    assert result is None


def test_validate_notification_1_rejects_missing_title():
    """
    Missing title should raise ValueError.
    """

    notification = {
        "description": "Continue engaging with CLAN.",
    }

    with pytest.raises(
        ValueError,
        match="title is missing",
    ):
        llm.validate_notification_1(notification)


def test_validate_notification_1_rejects_missing_description():
    """
    Missing description should raise ValueError.
    """

    notification = {
        "title": "Rahul, Keep Going",
    }

    with pytest.raises(
        ValueError,
        match="description is missing",
    ):
        llm.validate_notification_1(notification)


def test_validate_notification_1_rejects_empty_title():
    """
    Empty title should raise ValueError.
    """

    notification = {
        "title": "",
        "description": "Continue engaging with CLAN.",
    }

    with pytest.raises(
        ValueError,
        match="title is empty",
    ):
        llm.validate_notification_1(notification)


def test_validate_notification_1_rejects_empty_description():
    """
    Empty description should raise ValueError.
    """

    notification = {
        "title": "Rahul, Keep Going",
        "description": "",
    }

    with pytest.raises(
        ValueError,
        match="description is empty",
    ):
        llm.validate_notification_1(notification)


# ============================================================
# VALIDATE NOTIFICATION 2
# ============================================================

def test_validate_notification_2_accepts_valid_output():
    """
    Valid Notification 2 output should pass validation.
    """

    notification = {
        "title": "Rahul, Keep Growing",
        "description": (
            "Stay open to feedback and keep improving."
        ),
    }

    result = llm.validate_notification_2(notification)

    assert result is None


def test_validate_notification_2_rejects_missing_title():
    """
    Missing title should raise ValueError.
    """

    notification = {
        "description": "Stay open to feedback.",
    }

    with pytest.raises(
        ValueError,
        match="title is missing",
    ):
        llm.validate_notification_2(notification)


def test_validate_notification_2_rejects_missing_description():
    """
    Missing description should raise ValueError.
    """

    notification = {
        "title": "Rahul, Keep Growing",
    }

    with pytest.raises(
        ValueError,
        match="description is missing",
    ):
        llm.validate_notification_2(notification)


# ============================================================
# INVALID JSON
# ============================================================

def test_invalid_llm_json_raises_error(monkeypatch):
    """
    Verify invalid JSON returned by the LLM is rejected.
    """

    fake_client = FakeOpenAIClient(
        "This is not valid JSON"
    )

    monkeypatch.setattr(
        llm,
        "_get_openai_client",
        lambda: fake_client,
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        llm.generate_notification_1(
            user_name="Rahul",
            language="English",
            response_data={
                "questions_sent": 10,
                "questions_answered": 4,
                "response_percentage": 40,
                "notification_type": "IMPROVEMENT",
            },
        )


# ============================================================
# DIFFERENT Q&A → DIFFERENT LLM PROMPTS
# ============================================================

def test_different_answers_generate_different_prompts(
    monkeypatch,
):
    """
    Different selected answers must produce different
    prompts for the LLM.
    """

    prompts: list[str] = []

    class FakeLLMForComparison:
        def invoke(self, prompt: str):
            prompts.append(prompt)

            return type(
                "FakeResponse",
                (),
                {
                    "content": json.dumps(
                        {
                            "title": "Rahul, Keep Growing",
                            "description": (
                                "Use feedback to strengthen "
                                "your performance."
                            ),
                        }
                    )
                },
            )()

    # This test verifies prompt-building directly because
    # generate_notification_2 uses client.responses.create().
    prompt_1 = llm.build_notification_2_prompt(
        user_name="Rahul",
        language="English",
        prepared_qa={
            "responses": [
                {
                    "question": "How do you respond to feedback?",
                    "selected_answer": (
                        "I dismiss it if I disagree."
                    ),
                }
            ]
        },
    )

    prompt_2 = llm.build_notification_2_prompt(
        user_name="Rahul",
        language="English",
        prepared_qa={
            "responses": [
                {
                    "question": "How do you respond to feedback?",
                    "selected_answer": (
                        "I listen and make changes if needed."
                    ),
                }
            ]
        },
    )

    assert prompt_1 != prompt_2

    assert "I dismiss it if I disagree." in prompt_1
    assert "I listen and make changes if needed." in prompt_2


# ============================================================
# NOTIFICATION GENERATOR CLASS
# ============================================================

def test_notification_generator_class():
    """
    Verify the NotificationGenerator class can generate
    and validate a notification.
    """

    fake_output = {
        "title": "Rahul, Keep Improving",
        "description": (
            "Use CLAN insights to strengthen your performance."
        ),
        "action": "Watch now",
    }

    fake_client = FakeOpenAIClient(
        json.dumps(fake_output)
    )

    generator = llm.NotificationGenerator(
        client=fake_client,
        model="test-model",
    )

    result = generator.generate(
        "Generate a CLAN notification for Rahul."
    )

    assert result["title"] == "Rahul, Keep Improving"

    assert result["description"] == (
        "Use CLAN insights to strengthen your performance."
    )

    assert result["action"] == "Watch now"

    assert fake_client.responses.last_model == "test-model"
    assert fake_client.responses.last_input == (
        "Generate a CLAN notification for Rahul."
    )


# ============================================================
# NOTIFICATION GENERATOR CLASS — INVALID OUTPUT
# ============================================================

def test_notification_generator_rejects_invalid_output():
    """
    NotificationGenerator.validate() should reject invalid
    LLM output.
    """

    invalid_output = {
        "title": "",
        "description": "Some description",
    }

    with pytest.raises(ValueError):
        llm.NotificationGenerator.validate(
            invalid_output
        )