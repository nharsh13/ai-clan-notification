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

    kii_name = weakest_kii.get(
        "kii_name",
        "performance area",
    )

    video_title = video.get("title") or "learning video"

    return f"""
Generate ONE personalized AI-CLAN video notification.

User: {user_name}
Language: {language}
Area to improve: {kii_name}
Video: {video_title}

LANGUAGE:
- Use VERY SIMPLE, everyday language.
- Use common words and short sentences.
- Write naturally in the requested language.
- Be polite, warm, friendly, and respectful.
- Write for users who may understand only basic English or basic local-language words.
- Avoid difficult, formal, technical, professional, or complicated words.
- If a simpler word is possible, always use it.
- Do not use complicated motivational phrases.
- Do not use "please".
- Do not use emojis.

TITLE:
- Must start with "Hello {user_name}".
- Keep it short: 3–6 words.
- Make it positive and friendly.
- Do NOT include the improvement area "{kii_name}".
- Do NOT include the video title "{video_title}".
- Use simple everyday words.

DESCRIPTION:
- Keep it short: about 15–25 words.
- Use 2 short sentences.
- Mention ONLY the provided improvement area: {kii_name}.
- Encourage the user to watch the video.
- The notification opens the video when clicked, so it is okay to directly say "Watch this video".
- Keep the message positive and respectful.
- Clearly tell the user what to do.
- Do not make the user feel bad or guilty.

IMPORTANT VIDEO RULE:
- The user can click the notification and watch the selected video.
- Encourage the user to watch the video.
- Do NOT invent what the video teaches.
- Do NOT invent benefits of the video.
- Do NOT make claims about the video that are not provided.
- Do NOT include information that is not provided.

DO NOT:
- Mention performance percentages.
- Mention monthly targets.
- Mention daily targets.
- Mention seven-day targets.
- Mention question counts.
- Mention information that is not provided.
- Use difficult or formal language.
- Use technical or professional language.
- Use business jargon.
- Use negative or discouraging language.
- Shame, blame, or criticize the user.
- Use words such as "bad", "poor", "lazy", "weak", "failure",
  "lacking", "deficiency", or "underperforming".
- Add emojis.

FINAL CHECK:
- Is the title 3–6 simple words?
- Is the description about 15–25 words?
- Is the description only 2 short sentences?
- Is the improvement area mentioned?
- Does the description encourage the user to watch the video?
- Are all words simple and easy to understand?
- Is the message polite and friendly?
- Did I avoid making claims about the video?
- Is there only ONE main idea?

Return ONLY valid JSON with exactly these two fields:

{{
    "title": "string",
    "description": "string"
}}

No Markdown.
No explanation.
No additional fields.
""".strip()


def validate_performance_notification(
    notification: dict,
) -> None:
    """Validate the performance notification output."""

    if not isinstance(notification, dict):
        raise ValueError(
            "Performance notification must be a JSON object."
        )

    # The prompt requires exactly two fields.
    allowed_fields = {
        "title",
        "description",
    }

    extra_fields = set(notification.keys()) - allowed_fields

    if extra_fields:
        raise ValueError(
            "Performance notification contains unexpected fields: "
            f"{sorted(extra_fields)}"
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

    # Keep the notification suitable for mobile.
    if len(title.strip()) > 100:
        raise ValueError(
            "Performance notification title is too long."
        )

    if len(description.strip()) > 300:
        raise ValueError(
            "Performance notification description is too long."
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

    if not response:
        raise ValueError(
            "Performance notification generation returned no response."
        )

    content = getattr(response, "output_text", None)

    if not content:
        raise ValueError(
            "Performance notification generation returned no text."
        )

    content = content.strip()

    if not content:
        raise ValueError(
            "Performance notification generation returned empty text."
        )

    notification = _parse_json_response(content)

    validate_performance_notification(notification)

    return NotificationGenerator.validate(notification)