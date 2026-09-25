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
    answer_options: list[str] | None = None,
) -> str:
    """
    Build the prompt for the no-Q&A sentiment fallback notification.

    When there are no answers for the selected question:
    - Use the question to understand the situation.
    - Use the available answer options as possible helpful behaviors.
    - Select one relevant behavior.
    - Do not assume the user selected that behavior.
    """

    if question is not None:
        answer_options = answer_options or []

        answer_options_text = "\n".join(
            f"- {option}"
            for option in answer_options
            if isinstance(option, str) and option.strip()
        )

        if not answer_options_text:
            answer_options_text = "- No behavior options provided."

        return f"""
Generate ONE CLAN learning notification using the selected question and the available helpful behaviors below.

User: {user_name}
Language: {language}

SELECTED QUESTION:
{question}

AVAILABLE HELPFUL BEHAVIORS:
{answer_options_text}

REASONING:
- Understand the situation described by the selected question.
- Look at the available helpful behaviors.
- Choose ONE behavior that best fits the situation described by the question.
- Use the chosen behavior only as a helpful suggestion.
- Do NOT assume the user selected this behavior.
- Do NOT say the user already follows this behavior.
- Do NOT invent an answer to the question.
- Do NOT assume what the user feels, thinks, knows, or does.
- Do NOT claim the user learned something.
- Do NOT mention that the question has no answers.
- Do NOT mention missing data.
- Do NOT mention the available answer options in the notification.
- Do NOT give advice unrelated to the selected question.
- Do NOT create a new behavior that is not supported by the available behaviors.
- Use the word "customer".
- Never use "shopper", "shoppers", "buyer", or similar words.

LANGUAGE:
- Use VERY SIMPLE, everyday words and short sentences.
- Write naturally in the requested language.
- Be positive, helpful, warm, and respectful.
- Write for users who may understand only basic English or basic local-language words.
- Avoid difficult, formal, technical, professional, or business words.
- If a simpler word is possible, always use it.
- Do not use "please".
- Do not use emojis.
- Do not use complicated motivational phrases.

TITLE:
- Must start with "Hello {user_name},".
- Must include the user's name.
- Keep it short: 3–6 simple words.
- Make it positive and friendly.
- Do not mention the question.
- Do not mention that there is no answer.
- Do not use "great job" unless the question clearly supports it.

DESCRIPTION:
- Keep it around 15–25 words.
- Use 1–2 short sentences.
- Give ONE simple and useful tip.
- Base the tip on the selected question and ONE chosen behavior.
- Make the tip sound natural.
- Do NOT use labels such as:
  "Learning:"
  "Action:"
  "Tip:"
  "Suggestion:"
  "Advice:"
- Do NOT use headings.
- Do NOT use bullet points.
- Do NOT use a fixed format.
- Do NOT use a colon to create a label.
- Do not pressure, shame, blame, or criticize the user.

IMPORTANT APP BEHAVIOUR:
- Clicking the notification only opens the CLAN app.
- Do NOT tell the user to answer the question from the notification.
- Do NOT say that clicking the notification opens a specific question.
- Do NOT tell the user that they selected any answer.

FINAL CHECK:
- Is the notification based on the selected question?
- Did I choose ONE relevant behavior from the available behaviors?
- Did I avoid assuming the user selected that behavior?
- Is there only ONE clear tip?
- Is the language simple?
- Is the title 3–6 words?
- Is the description around 15–25 words?
- Does the description avoid "Learning:" and "Action:"?
- Does the notification avoid mentioning missing answers?
- Does the notification avoid mentioning the answer-option list?

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
    Build the prompt for the answered Q&A sentiment notification.
    """

    qa_json = json.dumps(
        prepared_qa,
        ensure_ascii=False,
    )

    return f"""
Generate ONE personalized CLAN learning notification.

User: {user_name}
Language: {language}

QUESTION AND RELATED ANSWERS:
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
- Always use the word "customer".
- Never use "shopper", "shoppers", "buyer", or similar words.

LEARNING RULES:
- Read all provided answers for the selected question together.
- Find ONE useful and positive learning point that is clearly supported by the answers.
- Give ONE simple suggestion based only on the provided answers.
- Do NOT create a new action that is not clearly supported by the answers.
- Do NOT invent facts about the user.
- Do NOT assume feelings, behavior, knowledge, or experience that is not shown in the answers.
- Combine the useful point and suggestion naturally into the notification.
- Do NOT use separate labels for the learning point or suggestion.
- Do NOT use a fixed format.
- Do NOT repeat the question.
- Do NOT repeat the answers.
- Do NOT summarize all the answers.
- Do NOT make unsupported claims.

DESCRIPTION:
- Write ONE natural notification message.
- Keep it short: about 12–20 words.
- Use 1–2 short sentences.
- Make the message sound like a normal notification.
- Give ONE useful insight or suggestion naturally.
- Do NOT use labels such as:
  "Learning:"
  "Action:"
  "Tip:"
  "Suggestion:"
  "Advice:"
- Do NOT use headings or bullet points.
- Do NOT use a colon to create a label.
- Do NOT separate the message into different sections.
- Keep the message positive and respectful.
- Do not make the user feel bad or guilty.
- Do not say the user's answer is wrong or bad.

TITLE:
- MUST start with "Hello {user_name},".
- MUST include the user's name.
- Keep it short: 3–6 words.
- Add a short, simple phrase after the user's name.
- Make it positive and friendly.
- Do not use difficult or formal words.
- Do not use complicated motivational phrases.
- Do not use "great job" unless the provided information clearly supports it.

IMPORTANT APP BEHAVIOUR:
- Clicking the notification only opens the CLAN app.
- Do NOT ask the user to answer a specific question from the notification.
- Do NOT mention a specific question in the notification.
- Do NOT tell the user to click the notification to answer a question.
- The notification should provide a useful tip based on the user's previous answer.

DO NOT:
- Use "Learning:".
- Use "Action:".
- Use "Tip:".
- Use "Suggestion:".
- Use "Advice:".
- Use labels or headings inside the description.
- Use bullet points.
- Use a fixed template.
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
- Does the title start with "Hello {user_name},"?
- Is the title 3–6 simple words?
- Is the description about 12–20 words?
- Is the description natural and conversational?
- Is there only ONE useful learning point?
- Is there only ONE useful suggestion?
- Is the suggestion based only on the provided answers?
- Does the description avoid "Learning:" and "Action:"?
- Does the description avoid all headings and labels?
- Is the language simple?
- Is the message polite and friendly?
- Does it avoid mentioning a specific question?
- Does it avoid saying the user can answer directly from the notification?
- Always use the word "customer".
- Never use "shopper", "shoppers", "buyer", or similar words.

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