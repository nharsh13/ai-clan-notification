from app.llm.llm_client import (
    MODEL,
    _generate_with_retry,
    _get_openai_client,
    _parse_json_response,
)


def build_engagement_sentiment_notification_prompt(
    user_name: str,
    language: str,
    response_data: dict,
) -> str:
    """
    Build the prompt for one personalized CLAN engagement notification.
    """

    notification_type = response_data.get(
        "notification_type",
        "IMPROVEMENT",
    )

    return f"""
Generate ONE personalized CLAN mobile notification.

User: {user_name}
Language: {language}
Type: {notification_type}

Rules:
- Write directly in the requested language.
- Use simple, friendly, everyday language.
- Keep the title short, meaningful, and encouraging.
- The title MUST include the user's name.
- The title MUST NOT be only "Hello {user_name},".
- Ensure the title and description are grammatically correct and natural in the requested language.

For IMPROVEMENT:
- Encourage the user to participate more in CLAN.
- Encourage the user to answer CLAN questions.
- Keep the message positive.

For POSITIVE:
- Appreciate the user's CLAN participation.
- Encourage the user to continue participating.

Do not:
- Mention response percentage.
- Mention question counts.
- Mention individual questions.
- Shame, blame, or criticize the user.
- Use words such as bad, poor, lazy, weak, or failure.
- Use difficult, formal, technical, or complicated language.
- Make unsupported claims.

The description must be short, clear, and actionable.

Return ONLY valid JSON with exactly these two fields:
{{
    "title": "string",
    "description": "string"
}}

No Markdown.
No explanation.
No additional fields.
""".strip()


def generate_engagement_sentiment_notification(
    user_name: str,
    language: str,
    response_data: dict,
) -> dict | None:
    """
    Generate one CLAN engagement notification using the configured LLM.

    Returns None when no response data is available.
    """

    if not response_data:
        return None

    client = _get_openai_client()

    prompt = build_engagement_sentiment_notification_prompt(
        user_name=user_name,
        language=language,
        response_data=response_data,
    )

    response = _generate_with_retry(
        client,
        MODEL,
        prompt,
    )

    content = response.output_text.strip()

    notification = _parse_json_response(content)

    validate_engagement_sentiment_notification(notification)

    return notification


def validate_engagement_sentiment_notification(
    notification: dict,
) -> None:
    """
    Validate the engagement sentiment notification output.
    """

    if not isinstance(notification, dict):
        raise ValueError(
            "Engagement sentiment notification must be a JSON object."
        )

    title = notification.get("title")
    description = notification.get("description")

    if not isinstance(title, str) or not title.strip():
        raise ValueError(
            "Engagement sentiment notification title must be a non-empty string."
        )

    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            "Engagement sentiment notification description must be a non-empty string."
        )