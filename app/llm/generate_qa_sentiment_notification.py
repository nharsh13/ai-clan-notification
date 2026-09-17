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
You are generating ONE personalized CLAN learning notification.

User name:
{user_name}

Notification language:
{language}

User's answered CLAN questions and responses:
{qa_json}

Instructions:

1. Generate EXACTLY ONE notification.
2. Analyze ALL of the user's answers together.
3. Identify one useful learning or improvement area from the answers.
4. Give the user one simple and positive suggestion based on that area.
5. Encourage the user to use this learning in their work.
6. Do NOT generate one notification for each question.
7. Do NOT directly repeat the question.
8. Do NOT directly repeat the user's answer.
9. Do NOT say that the user's answer is wrong or bad.
10. Do NOT shame, blame, or criticize the user.
11. Use very simple, everyday language.
12. Write like a short mobile app notification.
13. Do NOT use difficult, formal, technical, or complicated words.
14. The user's name MUST appear in the title.
15. The title MUST start with "Hello {user_name}".
16. The title must be short and meaningful.
17. Do NOT make the title only "Hello {user_name}".
18. Add a short motivational, encouraging, or useful phrase after the user's name.
19. Keep the description short and clear.
20. The description should give ONE simple and useful direction.
21. Generate the notification directly in the requested language.
22. Do NOT invent facts about the user.
23. Do NOT make claims that cannot be understood from the provided answers.
24. Return ONLY valid JSON.
25. Return exactly two fields:
    "title"
    "description"
26. Do NOT return Markdown, explanations, or any extra text.

For the title, use a natural and meaningful phrase such as:

- "Hello {user_name}, Keep Learning"
- "Hello {user_name}, Let's Grow"
- "Hello {user_name}, Keep Improving"
- "Hello {user_name}, Learn and Grow"
- "Hello {user_name}, Take the Next Step"
- "Hello {user_name}, Keep Moving Forward"
- "Hello {user_name}, Build Your Skills"
- "Hello {user_name}, Let's Get Better"
- "Hello {user_name}, Keep Growing"
- "Hello {user_name}, Your Next Step"

Choose ONE title that fits the learning area identified from the user's answers.
Do NOT always use the same title.

Example:

{{
    "title": "Hello {user_name}, Take the Next Step",
    "description": "Keep building your understanding and use what you learn in your work."
}}

Return ONLY:

{{
    "title": "string",
    "description": "string"
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

    response = _generate_with_retry(
        client,
        MODEL,
        prompt,
    )

    content = response.output_text.strip()

    notification = _parse_json_response(content)

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
