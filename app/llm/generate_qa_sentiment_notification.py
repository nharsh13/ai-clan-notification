import json

from app.llm.llm_client import (
    MODEL,
    _generate_with_retry,
    _get_openai_client,
    _parse_json_response,
)


def build_no_qa_sentiment_notification_prompt(
    user_name: str,
    language: str,
    question: str | None = None,
) -> str:
    """
    Build the prompt for the no-Q&A sentiment fallback notification.
    """

    if question is not None:
        return f"""
Generate ONE CLAN learning notification using only the selected question below.

User: {user_name}
Language: {language}
Selected question: {question}

REASONING:
- Understand the situation described by the selected question.
- Identify the positive, helpful behavior that fits that situation.
- Give ONE simple tip that encourages that behavior.
- Base the notification only on the selected question.
- Do NOT invent an answer to the question.
- Do NOT assume what the user feels, thinks, knows, or does.
- Do NOT say the user already follows the suggested behavior.
- Do NOT claim the user learned something.
- Do NOT mention answers, missing data, or that the question has no answers.
- Do NOT give advice unrelated to the question.

LANGUAGE:
- Use VERY SIMPLE, everyday words and short sentences.
- Write naturally in the requested language.
- Be positive, helpful, warm, and respectful.
- Avoid difficult, formal, technical, professional, or business words.
- Do not use "please".
- Do not use emojis or complicated motivational phrases.

TITLE:
- Must start with "Hello {user_name}," and include the user's name.
- Keep it short: 3–6 simple words.
- Make it positive and friendly.
- Do not mention the question in the title.

DESCRIPTION:
- Keep it around 15–25 words.
- Use 1–2 short sentences.
- Give ONE clear and useful behavior or tip.
- Keep it positive and practical. Do not pressure, shame, blame, or criticize the user.

IMPORTANT APP BEHAVIOUR:
- Clicking the notification only opens the CLAN app.
- Do NOT tell the user to answer the question from the notification.
- Do NOT say that clicking the notification opens a specific question.

FINAL CHECK:
- Is the notification based only on the selected question?
- Is the suggested behavior relevant, positive, and helpful?
- Did I avoid assumptions and invented information?
- Is there only ONE clear behavior or tip?
- Is the language simple, and are the title and description within the requested lengths?

Return ONLY valid JSON with exactly these two fields:

{{
    "title": "string",
    "description": "string"
}}

No Markdown.
No explanation.
No additional fields.
""".strip()

    return f"""
Generate ONE personalized CLAN engagement notification.

User: {user_name}
Language: {language}

RULES:
- Use VERY SIMPLE, everyday language.
- Use common words and short sentences.
- Write naturally in the requested language.
- Be polite, warm, friendly, and respectful.
- Write for users who may understand only basic English or basic local-language words.
- Avoid difficult, formal, technical, professional, or complicated words.
- If a simpler word is possible, always use it.
- Do not use "please".
- Do not use emojis.

TITLE:
- Must include the user's name.
- Keep it short: 3–6 words.
- Make it positive and friendly.
- Do not make it only "Hello {user_name}".
- Use simple everyday words.

DESCRIPTION:
- Keep it short: about 15–25 words.
- Use 2 short sentences.
- Encourage the user to keep using CLAN regularly.
- Encourage the user to share their thoughts and responses.
- Keep the message positive and supportive.
- Do not pressure the user.

IMPORTANT APP BEHAVIOUR:
- Clicking the notification only opens the CLAN app.
- Do NOT say or imply that the user can answer a question directly from the notification.
- Do NOT mention a specific question.
- Do NOT tell the user to click the notification to answer a question.
- The user can see and answer the daily question after opening CLAN.

DO NOT:
- Mention missing data.
- Mention no answers.
- Mention no Q&A.
- Mention internal system details.
- Mention debugging.
- Say that the user has no information or no records.
- Shame, blame, or criticize the user.
- Make the user feel guilty or pressured.
- Use difficult or formal language.
- Use technical or professional language.
- Make unsupported claims.
- Add information that is not provided.

FINAL CHECK:
- Is the title 3–6 simple words?
- Is the description about 15–25 words?
- Is the description only 2 short sentences?
- Is the message polite and friendly?
- Does it avoid mentioning a specific question?
- Does it avoid saying the user can answer directly from the notification?
- Is the language simple?

Return ONLY valid JSON with exactly these two fields:

{{
    "title": "string",
    "description": "string"
}}

No Markdown.
No explanation.
No additional fields.
""".strip()


def build_qa_sentiment_notification_prompt(
    user_name: str,
    language: str,
    prepared_qa: dict,
) -> str:
    """
    Build the prompt for the Q&A sentiment notification.
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

IMPORTANT LANGUAGE RULES:
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

LEARNING RULES:
- Read ALL provided answers together.
- Find ONE useful learning or improvement point.
- Give ONE simple suggestion based only on the answers.
- Give ONE clear action the user can try in their work.
- Use only information supported by the answers.
- Do NOT create one notification for each question.
- Do NOT repeat the questions.
- Do NOT repeat the answers.
- Do NOT summarize all the answers.
- Do NOT invent facts about the user.
- Do NOT make unsupported claims.

TITLE:
- MUST start with "Hello {user_name},".
- MUST include the user's name.
- Keep it short: 3–6 words.
- Add a short, simple phrase after the user's name.
- Make it positive and friendly.
- Do not use difficult or formal words.
- Do not use complicated motivational phrases.

DESCRIPTION:
- Keep it short: about 12–20 words.
- Use 1–2 short sentences.
- Give ONE useful suggestion.
- Give ONE clear action.
- Keep it positive and respectful.
- Do not make the user feel bad or guilty.
- Do not say the user's answer is wrong or bad.

IMPORTANT APP BEHAVIOUR:
- Clicking the notification only opens the CLAN app.
- Do NOT ask the user to answer a specific question from the notification.
- Do NOT mention a specific question.
- Do NOT tell the user to click the notification to answer a question.
- The notification should give a useful tip based on the user's previous answers.

DO NOT:
- Use difficult words.
- Use formal words.
- Use professional words.
- Use technical words.
- Use business jargon.
- Use long sentences.
- Use negative or discouraging language.
- Shame, blame, or criticize the user.
- Say the user's answer is wrong.
- Say the user's answer is bad.
- Invent facts about the user.
- Make claims not supported by the answers.
- Add information that is not provided.

FINAL CHECK:
- Is the title 3–6 simple words?
- Is the description about 12–20 words?
- Is there only ONE learning point?
- Is there only ONE useful action?
- Is the suggestion based only on the provided answers?
- Is the language simple?
- Is the message polite and friendly?
- Does it avoid mentioning a specific question?
- Does it avoid saying the user can answer directly from the notification?

Return ONLY valid JSON with exactly these two fields:

{{
    "title": "string",
    "description": "string"
}}

No Markdown.
No explanation.
No additional fields.
""".strip()


def generate_qa_sentiment_notification(
    user_name: str,
    language: str,
    prepared_qa: dict,
) -> dict:
    """
    Generate one Q&A sentiment notification using the configured LLM.
    """

    if not prepared_qa:
        raise ValueError(
            "Prepared Q&A data is required."
        )

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

    if not response:
        raise ValueError(
            "Q&A sentiment notification generation returned no response."
        )

    content = getattr(response, "output_text", None)

    if not content:
        raise ValueError(
            "Q&A sentiment notification generation returned no text."
        )

    content = content.strip()

    if not content:
        raise ValueError(
            "Q&A sentiment notification generation returned empty text."
        )

    notification = _parse_json_response(content)

    validate_qa_sentiment_notification(notification)

    return notification


def validate_qa_sentiment_notification(
    notification: dict,
) -> None:
    """
    Validate the Q&A sentiment notification output.
    """

    if not isinstance(notification, dict):
        raise ValueError(
            "Q&A sentiment notification must be a JSON object."
        )

    # The prompt requires exactly two fields.
    allowed_fields = {
        "title",
        "description",
    }

    extra_fields = set(notification.keys()) - allowed_fields

    if extra_fields:
        raise ValueError(
            "Q&A sentiment notification contains unexpected fields: "
            f"{sorted(extra_fields)}"
        )

    title = notification.get("title")
    description = notification.get("description")

    if not isinstance(title, str) or not title.strip():
        raise ValueError(
            "Q&A sentiment notification title "
            "must be a non-empty string."
        )

    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            "Q&A sentiment notification description "
            "must be a non-empty string."
        )

    # Keep the notification suitable for mobile.
    if len(title.strip()) > 100:
        raise ValueError(
            "Q&A sentiment notification title is too long."
        )

    if len(description.strip()) > 300:
        raise ValueError(
            "Q&A sentiment notification description is too long."
        )
