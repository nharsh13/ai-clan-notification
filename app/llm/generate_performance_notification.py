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
