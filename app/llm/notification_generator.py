import json

from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL

MODEL = OPENAI_MODEL


def build_performance_notification_prompt(
    user_name: str,
    language: str,
    weakest_kii: dict,
    video: dict,
) -> str:
    """Build a prompt for a notification tied to one selected video."""

    return f"""
You are generating ONE personalized AI-CLAN learning notification.

User name: {user_name}
Notification language: {language}
Weakest KII: {weakest_kii.get('kii_name', 'performance area')}
Current performance: {weakest_kii.get('performance_percentage', 0):.2f}%
Selected video title: {video.get('title') or 'Untitled learning video'}

Instructions:
1. Write exactly one positive, motivational, actionable notification.
2. Write it directly in the requested notification language.
3. Encourage the user to watch the selected video to improve the weakest KII.
4. Use only the weakest KII and selected video title supplied above.
5. Do not invent facts, techniques, outcomes, or video details.
6. Keep the title short and include the user's name.
7. Keep the description concise.
8. Return ONLY valid JSON with string keys "title", "description", and "action".
"""


# ============================================================
# NOTIFICATION 2
# ============================================================

def build_notification_2_prompt(
    user_name: str,
    language: str,
    prepared_qa: dict,
) -> str:
    """
    Build the prompt for Notification 2.
    """

    qa_json = json.dumps(
        prepared_qa,
        indent=2,
        ensure_ascii=False,
    )

    return f"""
You are generating ONE personalized workplace performance-improvement
notification for a CLAN user.

User name:
{user_name}

Notification language:
{language}

User's answered CLAN questions and responses:
{qa_json}

Instructions:

1. Analyze ALL Q&A together.
2. Understand the user's behavioral pattern.
3. Identify the strongest or most common improvement area.
4. Connect that improvement area to workplace performance.
5. Give a positive, practical and actionable direction.
6. Generate EXACTLY ONE notification for this user.
7. Do not generate one notification per question.
8. Do not shame, criticize, or call any answer wrong or bad.
9. Do not directly repeat the question or answer.
10. The user's name MUST appear in the title.
11. Keep the title short.
12. Keep the description concise.
13. Generate the notification directly in the requested language.
14. Return ONLY valid JSON.

Return:
{{
    "title": "Personalized title",
    "description": "Personalized description"
}}
"""


def generate_notification_2(
    user_name: str,
    language: str,
    prepared_qa: dict,
) -> dict:
    """
    Generate ONE Notification 2 using GPT-5 nano.
    """

    client = _get_openai_client()

    prompt = build_notification_2_prompt(
        user_name=user_name,
        language=language,
        prepared_qa=prepared_qa,
    )

    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    content = response.output_text.strip()

    notification = _parse_json_response(content)

    validate_notification_2(notification)

    return notification


def validate_notification_2(notification: dict) -> None:
    """
    Validate Notification 2 output.
    """

    if not isinstance(notification, dict):
        raise ValueError(
            "Notification 2 must be a JSON object."
        )

    if "title" not in notification:
        raise ValueError(
            "Notification 2 title is missing."
        )

    if "description" not in notification:
        raise ValueError(
            "Notification 2 description is missing."
        )

    if not isinstance(notification["title"], str):
        raise ValueError(
            "Notification 2 title must be a string."
        )

    if not isinstance(notification["description"], str):
        raise ValueError(
            "Notification 2 description must be a string."
        )

    if not notification["title"].strip():
        raise ValueError(
            "Notification 2 title is empty."
        )

    if not notification["description"].strip():
        raise ValueError(
            "Notification 2 description is empty."
        )


# ============================================================
# NOTIFICATION 1
# ============================================================

def build_notification_1_prompt(
    user_name: str,
    language: str,
    response_data: dict,
) -> str:
    """
    Build the prompt for Notification 1.

    Notification 1 uses only:
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


def generate_notification_1(
    user_name: str,
    language: str,
    response_data: dict,
) -> dict | None:
    """
    Generate ONE Notification 1 using GPT-5 nano.

    Returns None when no questions were sent.
    """

    if not response_data:
        return None

    client = _get_openai_client()

    prompt = build_notification_1_prompt(
        user_name=user_name,
        language=language,
        response_data=response_data,
    )

    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    content = response.output_text.strip()

    notification = _parse_json_response(content)

    validate_notification_1(notification)

    return notification


def validate_notification_1(notification: dict) -> None:
    """
    Validate Notification 1 output.
    """

    if not isinstance(notification, dict):
        raise ValueError(
            "Notification 1 must be a JSON object."
        )

    if "title" not in notification:
        raise ValueError(
            "Notification 1 title is missing."
        )

    if "description" not in notification:
        raise ValueError(
            "Notification 1 description is missing."
        )

    if not isinstance(notification["title"], str):
        raise ValueError(
            "Notification 1 title must be a string."
        )

    if not isinstance(notification["description"], str):
        raise ValueError(
            "Notification 1 description must be a string."
        )

    if not notification["title"].strip():
        raise ValueError(
            "Notification 1 title is empty."
        )

    if not notification["description"].strip():
        raise ValueError(
            "Notification 1 description is empty."
        )


# ============================================================
# SHARED HELPERS
# ============================================================

def _get_openai_client() -> OpenAI:
    """
    Create OpenAI client.
    """

    if not OPENAI_API_KEY:
        raise ValueError(
            "Missing required environment variable: OPENAI_API_KEY. "
            "Add it to the .env file."
        )

    return OpenAI(api_key=OPENAI_API_KEY)


def _parse_json_response(content: str) -> dict:
    """
    Parse JSON returned by the LLM.
    """

    try:
        return json.loads(content)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {content}"
        ) from exc


class NotificationGenerator:
    def __init__(self, client=None, model: str = MODEL):
        self.client = client
        self.model = model

    @staticmethod
    def validate(payload) -> dict[str, str]:
        if isinstance(payload, str):
            payload = _parse_json_response(payload)
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be an object")
        title = payload.get("title")
        description = payload.get("description")
        if not isinstance(title, str) or not title.strip() or not isinstance(description, str) or not description.strip():
            raise ValueError("LLM response requires non-empty title and description")
        action = payload.get("action", "Watch now")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("LLM response action must be a non-empty string")
        return {
            "title": title.strip(),
            "description": description.strip(),
            "action": action.strip(),
        }

    def generate(self, prompt: str) -> dict[str, str]:
        client = self.client or _get_openai_client()
        response = client.responses.create(model=self.model, input=prompt)
        return self.validate(response.output_text)
    
