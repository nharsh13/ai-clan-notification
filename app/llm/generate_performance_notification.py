from app.llm.llm_client import (
    MODEL,
    NotificationGenerator,
    _generate_with_retry,
    _get_openai_client,
    _parse_json_response,
)


def build_performance_notification_prompt(
    user_name: str,
    language: str,
    weakest_kii: dict,
    video: dict,
) -> str:
    """Build a short personalized AI-CLAN video recommendation prompt."""

    kii_name = weakest_kii.get("kii_name", "performance area")
    video_title = video.get("title") or "learning video"

    return f"""
Generate ONE personalized AI-CLAN video notification.

User: {user_name}
Language: {language}
Area to improve: {kii_name}
Video: {video_title}

Rules:
- Write directly in the requested language.
- Use simple, friendly, everyday language.
- Keep the notification short, positive, and motivating.
- Title MUST start with "Hello {user_name}".
- Keep the title short.
- Do NOT include the KII name or video title in the title.
- Mention the improvement area in the description.
- Encourage the user to watch the selected video.
- Do NOT mention performance percentages.
- Do NOT mention monthly, daily, or seven-day targets.
- Do NOT use difficult, formal, or technical language.
- Do NOT make negative or discouraging statements.
- Do NOT invent information about the video or its benefits.
- Mention only the provided improvement area.
- Return exactly two fields: "title" and "description".
- Return ONLY valid JSON.
- Do not return Markdown, explanations, or additional fields.

Example:
{{
    "title": "Hello {user_name}, Let's Get Better",
    "description": "{kii_name} is an area you can improve. Watch this video to learn and grow."
}}
""".strip()


def validate_performance_notification(
    notification: dict,
) -> None:
    """Validate the performance notification output."""

    if not isinstance(notification, dict):
        raise ValueError(
            "Performance notification must be a JSON object."
        )

    title = notification.get("title")
    description = notification.get("description")

    if not isinstance(title, str) or not title.strip():
        raise ValueError(
            "Performance notification title must be a non-empty string."
        )

    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            "Performance notification description must be a non-empty string."
        )

    action = notification.get("action", "Watch now")

    if not isinstance(action, str) or not action.strip():
        raise ValueError(
            "Performance notification action must be a non-empty string."
        )


def generate_performance_notification(
    user_name: str,
    language: str,
    weakest_kii: dict,
    video: dict,
) -> dict:
    """Generate the performance recommendation notification."""

    client = _get_openai_client()

    prompt = build_performance_notification_prompt(
        user_name=user_name,
        language=language,
        weakest_kii=weakest_kii,
        video=video,
    )

    response = _generate_with_retry(
        client,
        MODEL,
        prompt,
    )

    notification = _parse_json_response(
        response.output_text.strip()
    )

    validate_performance_notification(notification)

    return NotificationGenerator.validate(notification)