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
    """Build a simple personalized video recommendation notification."""

    return f"""
Generate ONE personalized AI-CLAN video notification.

User name: {user_name}
Notification language: {language}
Weakest KII: {weakest_kii.get('kii_name', 'performance area')}
Selected video title: {video.get('title') or 'Untitled learning video'}

Rules:

1. Write the notification in the requested language.
2. Use very simple English/words that are easy for any app user to understand.
3. Keep the notification short, clear, positive, and motivating.
4. The title must start with "Hello {user_name}".
5. Keep the title very short.
6. Do NOT include the KII name in the title.
7. Do NOT include the video title in the title.
8. Use a simple motivational title such as:
   - "Hello {user_name}, Let's Improve"
   - "Hello {user_name}, Keep Growing"
   - "Hello {user_name}, Let's Grow"
   - "Hello {user_name}, Keep Learning"
9. In the description, mention the weakest KII as an area the user can improve.
10. Encourage the user to watch the selected video.
11. Do NOT mention the exact performance percentage.
12. Do NOT mention monthly target, daily target, or seven-day target.
13. Do NOT use difficult, formal, or complicated words.
14. Do NOT make negative or discouraging statements.
15. Do NOT invent information about the video or what the user will achieve from it.
16. Do NOT mention any KII other than the weakest KII.
17. Do NOT generate user_id, notification_type, reference_id, video_popup, image, success, or any other fields.
18. Return ONLY valid JSON.
19. Return exactly two fields: "title" and "description".

Example output:

{{
  "title": "Hello Veera, Let's Improve",
  "description": "Channel Partner Empanelled is an area you can improve. Watch this video to learn and grow."
}}
"""

def validate_performance_notification(notification: dict) -> None:
    """Validate the performance notification output."""
    if not isinstance(notification, dict):
        raise ValueError("Performance notification must be a JSON object.")
    if "title" not in notification:
        raise ValueError("Performance notification title is missing.")
    if "description" not in notification:
        raise ValueError("Performance notification description is missing.")
    if not isinstance(notification["title"], str) or not notification["title"].strip():
        raise ValueError("Performance notification title is empty.")
    if not isinstance(notification["description"], str) or not notification["description"].strip():
        raise ValueError("Performance notification description is empty.")
    action = notification.get("action", "Watch now")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("Performance notification action must be a non-empty string")


def generate_performance_notification(
    user_name: str,
    language: str,
    weakest_kii: dict,
    video: dict,
) -> dict:
    """Generate the performance recommendation notification using the configured LLM."""
    client = _get_openai_client()
    prompt = build_performance_notification_prompt(
        user_name=user_name,
        language=language,
        weakest_kii=weakest_kii,
        video=video,
    )
    response = _generate_with_retry(client, MODEL, prompt)
    notification = _parse_json_response(response.output_text.strip())
    validate_performance_notification(notification)
    return NotificationGenerator.validate(notification)
