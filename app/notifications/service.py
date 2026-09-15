from __future__ import annotations

from typing import Any

from app.config import OPENAI_API_KEY
from app.database.connection import engine
from app.database.user_repository import get_user
from app.llm.notification_generator import NotificationGenerator, build_notification_1_prompt, build_notification_2_prompt
from app.notifications.engine import NotificationEngine
from app.notifications.models import NotificationRequest, NotificationResponse
from app.notifications.sender import NotificationSender
from app.performance.performance import calculate_performance
from app.recommendation.recommendation import recommend_video
from app.sentiment.sentiment import (
    get_next_sentiment_response,
    get_responses,
    get_user_response_rate,
    save_sentiment_notification_history,
)


class NotificationService:
    """Coordinate the AI-CLAN notification pipeline."""

    def __init__(
        self,
        sender: NotificationSender | None = None,
        generator: NotificationGenerator | None = None,
        db_engine=engine,
    ) -> None:
        self.sender = sender or NotificationSender()
        self.generator = generator or NotificationGenerator()
        self.db_engine = db_engine
        self.engine = NotificationEngine()

    def _get_user_profile(self, user_id: int) -> dict[str, Any]:
        profile = get_user(user_id, self.db_engine)
        if profile is None:
            return {
                "user_name": f"User {user_id}",
                "language_code": "en",
                "video_language_id": None,
            }
        return {
            "user_name": profile.get("user_name") or profile.get("name") or f"User {user_id}",
            "language_code": str(profile.get("language_code") or "en").strip().lower() or "en",
            "video_language_id": profile.get("video_language_id"),
        }

    def _fallback_notification(self, flow: str, user_name: str, context: str) -> dict[str, str]:
        if flow == "engagement":
            return {
                "title": f"{user_name}, Keep Building Momentum",
                "description": f"Your CLAN engagement matters. Keep participating so your progress stays strong and consistent.{context}",
            }
        if flow == "sentiment":
            return {
                "title": f"{user_name}, Your Progress Matters",
                "description": f"Your recent responses show a clear opportunity to keep growing with practical next steps.{context}",
            }
        return {
            "title": f"{user_name}, Focus on {context}",
            "description": "Keep refining the area that is currently your biggest opportunity and build momentum with small, consistent actions.",
        }

    def _generate_llm_notification(self, flow: str, user_name: str, payload: dict[str, Any]) -> dict[str, str]:
        try:
            if flow == "engagement":
                prompt = build_notification_1_prompt(
                    user_name=user_name,
                    language=payload.get("language", "English"),
                    response_data=payload["response_data"],
                )
            elif flow == "sentiment":
                prompt = build_notification_2_prompt(
                    user_name=user_name,
                    language=payload.get("language", "English"),
                    prepared_qa=payload["prepared_qa"],
                )
            else:
                weakest = payload["weakest_kii"]
                video_title = payload.get("video_title") or "the right learning resource"
                prompt = (
                    "You are generating a personalized workplace performance notification. "
                    f"Return valid JSON with keys title and description. "
                    f"User name: {user_name}. "
                    f"Language: {payload.get('language', 'English')}. "
                    f"Current weakest area: {weakest.get('kii_name', 'performance area')} with performance {weakest.get('performance_percentage', 0):.2f}%. "
                    f"Suggested video: {video_title}. "
                    "Keep the title short and personal, and keep the description specific, encouraging, and actionable. "
                )
            return self.generator.generate(prompt)
        except Exception:
            return self._fallback_notification(flow, user_name, payload.get("context", "your current growth area"))

    def _build_flow_context(self, request: NotificationRequest) -> tuple[str, dict[str, Any]]:
        profile = self._get_user_profile(request.user_id)
        user_name = str(profile["user_name"]).strip() or f"User {request.user_id}"
        language = str(profile["language_code"]).strip() or "en"
        if request.flow == "engagement":
            response_data = get_user_response_rate(request.user_id)
            return user_name, {
                "language": language,
                "response_data": response_data or {"questions_sent": 0, "questions_answered": 0, "response_percentage": 0, "notification_type": "IMPROVEMENT"},
                "context": "your CLAN participation",
            }
        if request.flow == "sentiment":
            prepared_qa = prepare_user_qa(request.user_id)
            return user_name, {
                "language": language,
                "prepared_qa": prepared_qa,
                "context": "your recent responses and workplace reflections",
            }

        # performance flow
        calc = calculate_performance(request.user_id)
        weakest = calc["improvement_area"]
        language_id = profile.get("video_language_id")
        recommendation = None
        try:
            performance = type("Perf", (), {"kii_name": weakest["kii_name"], "kii_id": weakest["kii_id"]})()
            recommendation = recommend_video(
                performance,
                language_id,
                lambda _: [0.0] * 384,
                self.db_engine,
                user_id=request.user_id,
            )
        except Exception:
            recommendation = None

        return user_name, {
            "language": language,
            "weakest_kii": weakest,
            "video_title": (recommendation or {}).get("title") if isinstance(recommendation, dict) else None,
            "video_id": (recommendation or {}).get("video_id") if isinstance(recommendation, dict) else None,
            "context": weakest.get("kii_name", "your current growth area"),
        }

    def get_performance(self, user_id: int) -> dict[str, Any]:
        result = calculate_performance(user_id)
        return {
            "user_id": user_id,
            "seven_day_performance": result["performance"],
            "weakest_kii": result["improvement_area"],
        }

    def get_engagement(self, user_id: int) -> dict[str, Any]:
        response_data = get_user_response_rate(user_id)
        if response_data is None:
            return {
                "user_id": user_id,
                "questions_sent": 0,
                "questions_answered": 0,
                "response_percentage": 0.0,
                "notification_type": "IMPROVEMENT",
            }
        return response_data

    def get_sentiment(self, user_id: int) -> dict[str, Any]:
        responses = get_responses(user_id, self.db_engine)
        return {"user_id": user_id, "responses": responses}

    def process_sentiment_notification(self, user_id: int) -> dict[str, Any]:
        response = get_next_sentiment_response(user_id, self.db_engine)
        if response is None:
            return {"user_id": user_id, "notification": None}

        result = {
            "user_id": user_id,
            "notification": {
                "response_id": response["response_id"],
                "question_id": response["question_id"],
                "answer_id": response["answer_id"],
                "question": response["question"],
                "answer": response["answer"],
            },
        }

        if self.sender.remote_url:
            remote_response = self.sender.send(
                user_id=user_id,
                notification_type="sentiment",
                title=response["question"],
                description=response["answer"],
                reference_id=int(response["response_id"]),
                video_popup="N",
            )
            save_sentiment_notification_history(response, self.db_engine)
            result["remote_send_status"] = "sent"
            result["remote_send_response"] = remote_response
        else:
            result["remote_send_status"] = "skipped"
            result["error"] = "REMOTE_NOTIFICATION_SEND_URL is not configured"

        return result

    def build_notification(self, request: NotificationRequest) -> NotificationResponse:
        if request.user_id <= 0:
            raise ValueError("user_id must be greater than zero")

        user_name, payload = self._build_flow_context(request)
        notification_data = self._generate_llm_notification(request.flow, user_name, payload)

        video_id = request.video_id or payload.get("video_id")
        video_title = request.video_title or payload.get("video_title")
        creator_name = request.creator_name
        deep_link = request.deep_link

        notification = self.engine.make_notification(
            user_id=request.user_id,
            flow=request.flow,
            notification_type=request.notification_type or request.flow,
            title=notification_data["title"],
            description=notification_data["description"],
            reference_id=request.reference_id or video_id,
            deep_link=deep_link,
            video_id=video_id,
            video_title=video_title,
            creator_name=creator_name,
            should_send=request.should_send,
            video_popup=request.video_popup,
        )

        response = NotificationResponse(
            user_id=request.user_id,
            flow=request.flow,
            campaign_day=request.campaign_day,
            notification_title=notification.notification_title,
            notification_body=notification.notification_body,
            audience_strategy=notification.audience_strategy,
            cohort_key=notification.cohort_key,
            action=notification.action,
            deep_link=notification.deep_link,
            notification_type=notification.notification_type,
            should_send=request.should_send,
            reference_id=notification.reference_id,
            video_id=notification.video_id,
            video_title=notification.video_title,
            creator_name=notification.creator_name,
            video_popup=notification.video_popup,
        )

        if request.should_send:
            if not self.sender.remote_url:
                response.remote_send_status = "skipped"
                response.error = "REMOTE_NOTIFICATION_SEND_URL is not configured"
            else:
                try:
                    remote_response = self.sender.send(
                        user_id=request.user_id,
                        notification_type=notification.notification_type or request.flow,
                        title=notification.notification_title,
                        description=notification.notification_body,
                        reference_id=int(notification.reference_id or 0),
                        video_popup=notification.video_popup or "N",
                    )
                    response.remote_send_status = "sent"
                    response.remote_send_response = remote_response
                except Exception as exc:  # pragma: no cover - defensive fallback
                    response.remote_send_status = "failed"
                    response.error = str(exc)

        return response


__all__ = ["NotificationService"]
