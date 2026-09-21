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
        ensure_ascii=False,
    )

    return f"""
Generate ONE personalized CLAN learning notification.

User: {user_name}
Language: {language}

Answered CLAN questions and responses:
{qa_json}

Requirements:
- Analyze ALL answers together.
- Identify ONE useful learning or improvement area.
- Give ONE simple, positive suggestion based on that area.
- Encourage the user to apply the learning in their work.
- Do not generate one notification per question.
- Ensure the title and description are grammatically correct and natural in the requested language.
- Do not directly repeat questions or answers.
- Do not say the user's answer is wrong or bad.
- Do not shame, blame, or criticize the user.
- Use simple, friendly, everyday language.
- Write like a short mobile app notification.
- Do not use difficult, formal, technical, or complicated words.
- Title MUST start with "Hello {user_name},".
- Title MUST include the user's name.
- Title MUST NOT be only "Hello {user_name},".
- Keep the title short and meaningful.
- Add a short motivational or useful phrase after the user's name.
- Description must be short, clear, and actionable.
- Give only ONE useful direction.
- Write directly in the requested language.
- Do not invent facts about the user.
- Do not make claims unsupported by the provided answers.

Return ONLY valid JSON with exactly these two fields:
{{
    "title": "string",
    "description": "string"
}}

Do not return Markdown, explanations, or additional fields.
""".strip()


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

    response = _generate_with_retry(
        client,
        MODEL,
        prompt,
    )

    notification = _parse_json_response(
        response.output_text.strip()
    )

    validate_qa_sentiment_notification(notification)

    return notification


def validate_qa_sentiment_notification(
    notification: dict,
) -> None:
    """
    Validate the Q/A sentiment notification output.
    """

    if not isinstance(notification, dict):
        raise ValueError(
            "Q/A sentiment notification must be a JSON object."
        )

    title = notification.get("title")
    description = notification.get("description")

    if not isinstance(title, str) or not title.strip():
        raise ValueError(
            "Q/A sentiment notification title must be a non-empty string."
        )

    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            "Q/A sentiment notification description must be a non-empty string."
        )