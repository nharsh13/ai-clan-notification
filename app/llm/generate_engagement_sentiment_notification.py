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
    Build the prompt for the engagement sentiment notification.

    This notification uses only:
        - user name
        - questions sent
        - questions answered
        - response percentage
        - notification type

    It does NOT use question or answer text.
    """

    questions_sent = response_data["questions_sent"]
    questions_answered = response_data["questions_answered"]
    response_percentage = response_data["response_percentage"]
    notification_type = response_data["notification_type"]

    return f"""
You are generating ONE personalized CLAN engagement notification.

User name:
{user_name}

Notification language:
{language}

Questions sent:
{questions_sent}

Questions answered:
{questions_answered}

Response percentage:
{response_percentage}%

Notification type:
{notification_type}

Instructions:

1. Generate EXACTLY ONE notification.
2. The notification is about CLAN engagement.
3. Use the response percentage and notification type.
4. If the type is IMPROVEMENT, encourage the user positively
   to participate more in CLAN.
5. If the type is POSITIVE, appreciate their CLAN participation
   and encourage them to continue.
6. Never shame or criticize the user.
7. Do not use negative words such as bad, poor, lazy, or similar wording.
8. Do not mention individual questions or answers.
9. The user's name MUST appear in the title.
10. Keep the title short.
11. Keep the description concise and actionable.
12. Generate the notification directly in the requested language.
13. Return ONLY valid JSON.

Return:
{{
    "title": "Personalized title",
    "description": "Personalized description"
}}
"""


def generate_engagement_sentiment_notification(
    user_name: str,
    language: str,
    response_data: dict,
) -> dict | None:
    """
    Generate one engagement sentiment notification using the configured LLM.

    Returns None when no questions were sent.
    """

    if not response_data:
        return None

    client = _get_openai_client()

    prompt = build_engagement_sentiment_notification_prompt(
        user_name=user_name,
        language=language,
        response_data=response_data,
    )

    response = _generate_with_retry(client, MODEL, prompt)

    content = response.output_text.strip()

    notification = _parse_json_response(content)

    validate_engagement_sentiment_notification(notification)

    return notification


def validate_engagement_sentiment_notification(notification: dict) -> None:
    """
    Validate the engagement sentiment notification output.
    """

    if not isinstance(notification, dict):
        raise ValueError(
            "Engagement sentiment notification must be a JSON object."
        )

    if "title" not in notification:
        raise ValueError(
            "Engagement sentiment notification title is missing."
        )

    if "description" not in notification:
        raise ValueError(
            "Engagement sentiment notification description is missing."
        )

    if not isinstance(notification["title"], str):
        raise ValueError(
            "Engagement sentiment notification title must be a string."
        )

    if not isinstance(notification["description"], str):
        raise ValueError(
            "Engagement sentiment notification description must be a string."
        )

    if not notification["title"].strip():
        raise ValueError(
            "Engagement sentiment notification title is empty."
        )

    if not notification["description"].strip():
        raise ValueError(
            "Engagement sentiment notification description is empty."
        )
