import json

from app.llm.llm_client import (
    MODEL,
    _generate_with_retry,
    _get_openai_client,
    _parse_json_response,
)


def build_qa_sentiment_notification_prompt(
    user_name: str,
    language: str,
    prepared_qa: dict,
) -> str:
    """
    Build the prompt for the Q/A sentiment notification.
    """

    qa_json = json.dumps(
        prepared_qa,
        indent=2,
        ensure_ascii=False,
    )

    return f"""
You are generating ONE personalized workplace performance-improvement
notification for a CLAN user.

User name:
{user_name}

Notification language:
{language}

User's answered CLAN questions and responses:
{qa_json}

Instructions:

1. Analyze ALL Q&A together.
2. Understand the user's behavioral pattern.
3. Identify the strongest or most common improvement area.
4. Connect that improvement area to workplace performance.
5. Give a positive, practical and actionable direction.
6. Generate EXACTLY ONE notification for this user.
7. Do not generate one notification per question.
8. Do not shame, criticize, or call any answer wrong or bad.
9. Do not directly repeat the question or answer.
10. The user's name MUST appear in the title.
11. Keep the title short.
12. Keep the description concise.
13. Generate the notification directly in the requested language.
14. Return ONLY valid JSON.

Return:
{{
    "title": "Personalized title",
    "description": "Personalized description"
}}
"""


def generate_qa_sentiment_notification(
    user_name: str,
    language: str,
    prepared_qa: dict,
) -> dict:
    """
    Generate one Q/A sentiment notification using the configured LLM.
    """

    client = _get_openai_client()

    prompt = build_qa_sentiment_notification_prompt(
        user_name=user_name,
        language=language,
        prepared_qa=prepared_qa,
    )

    response = _generate_with_retry(client, MODEL, prompt)

    content = response.output_text.strip()

    notification = _parse_json_response(content)

    validate_qa_sentiment_notification(notification)

    return notification


def validate_qa_sentiment_notification(notification: dict) -> None:
    """
    Validate the Q/A sentiment notification output.
    """

    if not isinstance(notification, dict):
        raise ValueError(
            "Q/A sentiment notification must be a JSON object."
        )

    if "title" not in notification:
        raise ValueError(
            "Q/A sentiment notification title is missing."
        )

    if "description" not in notification:
        raise ValueError(
            "Q/A sentiment notification description is missing."
        )

    if not isinstance(notification["title"], str):
        raise ValueError(
            "Q/A sentiment notification title must be a string."
        )

    if not isinstance(notification["description"], str):
        raise ValueError(
            "Q/A sentiment notification description must be a string."
        )

    if not notification["title"].strip():
        raise ValueError(
            "Q/A sentiment notification title is empty."
        )

    if not notification["description"].strip():
        raise ValueError(
            "Q/A sentiment notification description is empty."
        )
