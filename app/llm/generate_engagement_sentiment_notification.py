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
- Must include the user's name.
- Keep it short: 3–6 words.
- Make it positive and friendly.
- Do NOT make it only "Hello {user_name}".
- Use simple everyday words.

DESCRIPTION:
- Keep it medium-short: about 15–25 words.
- Use 2 short sentences.
- Keep it polite and natural.
- Encourage the user to keep using CLAN regularly.
- Encourage the user to share their thoughts or responses.
- Make the message positive and supportive.

IMPORTANT APP BEHAVIOUR:
- Clicking the notification only opens the CLAN app.
- Do NOT say or imply that the user can answer a question directly from the notification.
- Do NOT tell the user to click the notification to answer a question.
- Do NOT mention a specific question.
- Do NOT describe or refer to an individual question.
- The user can see and answer the daily question after opening CLAN.

FOR IMPROVEMENT:
- Encourage the user to use CLAN regularly.
- Encourage the user to share their thoughts and responses.
- Keep the message positive.
- Do not make the user feel bad or guilty.

FOR POSITIVE:
- Appreciate the user's participation in CLAN.
- Encourage the user to continue using CLAN.
- Encourage them to keep sharing their thoughts and responses.

DO NOT:
- Mention response percentage.
- Mention question counts.
- Mention individual questions.
- Mention missing data.
- Mention internal system details.
- Mention debugging.
- Shame, blame, or criticize the user.
- Make the user feel guilty or pressured.
- Use words such as "bad", "poor", "lazy", "weak", or "failure".
- Use difficult or formal language.
- Use technical or professional language.
- Make unsupported claims.
- Add information that is not provided.

FINAL CHECK:
- Is the title 3–6 simple words?
- Is the description about 15–25 words?
- Is the description only 2 short sentences?
- Is the language easy to understand?
- Is the message polite and friendly?
- Does it avoid telling the user they can answer directly from the notification?
- Does it avoid mentioning a specific question?
- Is there only one main message?


FACTUAL ACCURACY:
- Use ONLY information explicitly provided in the input data.
- Do NOT invent user activities.
- Do NOT say the user created posts unless the input explicitly says they created posts.
- Do NOT say the user shared posts unless the input explicitly says they shared posts.
- Do NOT say the user commented unless the input explicitly says they commented.
- Do NOT say the user joined discussions unless the input explicitly says so.
- Do NOT say the user liked, reacted, watched, answered, or shared anything unless the input explicitly supports it.
- Do NOT use words such as "posts", "posting", "shared", "commented", "joined in", or "participated" unless supported by the input.
- When there is no specific activity available, use a general encouragement message instead.


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

    if not response:
        return None

    content = getattr(response, "output_text", None)

    if not content:
        return None

    content = content.strip()

    if not content:
        return None

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

    # The prompt requires exactly two fields.
    allowed_fields = {
        "title",
        "description",
    }

    extra_fields = set(notification.keys()) - allowed_fields

    if extra_fields:
        raise ValueError(
            "Engagement sentiment notification contains unexpected fields: "
            f"{sorted(extra_fields)}"
        )

    title = notification.get("title")
    description = notification.get("description")

    if not isinstance(title, str) or not title.strip():
        raise ValueError(
            "Engagement sentiment notification title "
            "must be a non-empty string."
        )

    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            "Engagement sentiment notification description "
            "must be a non-empty string."
        )

    # Keep the notification suitable for mobile.
    if len(title.strip()) > 100:
        raise ValueError(
            "Engagement sentiment notification title is too long."
        )

    if len(description.strip()) > 300:
        raise ValueError(
            "Engagement sentiment notification description is too long."
        )