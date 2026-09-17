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

    questions_sent = response_data.get("questions_sent", 0)
    questions_answered = response_data.get("questions_answered", 0)
    response_percentage = response_data.get("response_percentage", 0)
    notification_type = response_data.get(
        "notification_type",
        "IMPROVEMENT",
    )

    return f"""
You are generating ONE personalized CLAN engagement notification.

User name: {user_name}
Notification language: {language}
Questions sent: {questions_sent}
Questions answered: {questions_answered}
Response percentage: {response_percentage}%
Notification type: {notification_type}

Your goal:
Create a short, friendly notification that encourages the user to stay engaged with CLAN.

Rules:

1. Generate EXACTLY ONE notification.
2. Write the notification directly in the requested language.
3. Use very simple, everyday language.
4. Write like a mobile app notification.
5. Keep the title short, interesting, and meaningful.
6. The title MUST include the user's name.
7. The title MUST NOT be only "Hello {user_name}".
8. The title should contain a short motivational or encouraging phrase.

9. If notification_type is "IMPROVEMENT":
   - Encourage the user to participate more in CLAN.
   - Encourage the user to answer more CLAN questions.
   - Keep the message positive.
   - Use an encouraging title such as:
     - "Hello {user_name}, Let's Stay Engaged"
     - "Hello {user_name}, Your Voice Matters"
     - "Hello {user_name}, Let's Keep Growing"
     - "Hello {user_name}, Keep Taking Part"
     - "Hello {user_name}, Let's Keep Moving"
     - "Hello {user_name}, Keep Building"
     - "Hello {user_name}, Stay Connected"
     - "Hello {user_name}, Keep Going"

10. If notification_type is "POSITIVE":
    - Appreciate the user's CLAN participation.
    - Encourage the user to continue participating.
    - Use a positive title such as:
      - "Hello {user_name}, Great Progress"
      - "Hello {user_name}, Great Work"
      - "Hello {user_name}, Keep Learning"
      - "Hello {user_name}, Keep Growing"
      - "Hello {user_name}, Keep It Going"
      - "Hello {user_name}, You're Doing Great"
      - "Hello {user_name}, Keep Taking Part"

11. Do NOT show the response percentage.
12. Do NOT show the number of questions sent or answered.
13. Do NOT mention individual questions.
14. Do NOT shame, blame, or criticize the user.
15. Do NOT use words such as:
    "bad", "poor", "lazy", "weak", "failure",
    or similar negative words.
16. Do NOT use difficult, formal, technical, or complicated words.
17. Do NOT make claims that are not supported by the input.
18. Keep the description short, clear, and actionable.
19. The description should tell the user what they can do next.
20. Return ONLY valid JSON.
21. Return exactly two fields:
    "title"
    "description"
22. Do NOT return Markdown, explanations, or any extra text.

Example for IMPROVEMENT:

{{
    "title": "Hello {user_name}, Let's Stay Engaged",
    "description": "Answer more CLAN questions and keep learning."
}}

Example for POSITIVE:

{{
    "title": "Hello {user_name}, Great Progress",
    "description": "Keep taking part in CLAN and continue learning."
}}

Return ONLY:

{{
    "title": "string",
    "description": "string"
}}
"""


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
