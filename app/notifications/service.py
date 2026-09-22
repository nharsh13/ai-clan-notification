from __future__ import annotations

from typing import Any
import logging

from sqlalchemy import text

from app.config import OPENAI_API_KEY
from app.database.connection import engine
from app.database.notification_repository import get_next_manual_notification_for_user
from app.database.user_repository import get_user
from app.llm.generate_engagement_sentiment_notification import (
    build_engagement_sentiment_notification_prompt,
)
from app.llm.generate_performance_notification import (
    build_performance_notification_prompt,
)
from app.llm.generate_qa_sentiment_notification import (
    build_qa_sentiment_notification_prompt,
)
from app.llm.llm_client import NotificationGenerator
from app.notifications.engine import NotificationEngine
from app.notifications.models import NotificationProcessingResult, NotificationRequest
from app.notifications.sender import NotificationSender
from app.constants import FLOW_BY_EVENT_TYPE
from app.performance.performance import calculate_performance
from app.recommendation.recommendation import embed_text, recommend_video
from app.sentiment.sentiment import (
    get_eligible_user_qa,
    get_user_response_rate,
    prepare_user_qa,
    save_sentiment_notification_history,
)

logger = logging.getLogger(__name__)

NO_ENGAGEMENT_DATA = "NO_ENGAGEMENT_DATA"
NO_QA_DATA = "NO_QA_DATA"
NO_VIDEO_RECOMMENDATION = "NO_VIDEO_RECOMMENDATION"
LLM_NO_RESPONSE = "LLM_NO_RESPONSE"
MISSING_REQUIRED_DATA = "MISSING_REQUIRED_DATA"


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
        self.last_skip_reason: str | None = None

    def _get_user_profile(
        self,
        user_id: int,
        require_video_language: bool = False,
    ) -> dict[str, Any]:
        profile = get_user(user_id, self.db_engine)
        if profile is None:
            self.last_skip_reason = MISSING_REQUIRED_DATA
            return None

        language_code = str(profile.get("app_language_code") or "").strip().lower()
        if not language_code:
            self.last_skip_reason = MISSING_REQUIRED_DATA
            return None
        video_language_ids = profile.get("video_language_ids") or []

        return {
            "user_name": profile.get("user_name") or profile.get("name") or f"User {user_id}",
            "app_language_code": language_code,
            "video_language_ids": [int(value) for value in video_language_ids],
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

    def _get_telugu_video_language_id(self) -> int | None:
        query = text(
            """
            SELECT id
            FROM public.md_language
            WHERE lower(code) = 'te'
               OR lower(name) = 'telugu'
            ORDER BY CASE WHEN lower(code) = 'te' THEN 0 ELSE 1 END, id
            LIMIT 1
            """
        )
        with self.db_engine.connect() as connection:
            language_id = connection.execute(query).scalar_one_or_none()
        return int(language_id) if language_id is not None else None

    def _generate_llm_notification(self, flow: str, user_name: str, payload: dict[str, Any]) -> dict[str, str]:
        if flow == "engagement":
            prompt = build_engagement_sentiment_notification_prompt(
                user_name=user_name,
                language=payload.get("language", "English"),
                response_data=payload["response_data"],
            )
        elif flow == "sentiment":
            prompt = build_qa_sentiment_notification_prompt(
                user_name=user_name,
                language=payload.get("language", "English"),
                prepared_qa=payload["prepared_qa"],
            )
        else:
            prompt = build_performance_notification_prompt(
                user_name=user_name,
                language=payload["language"],
                weakest_kii=payload["weakest_kii"],
                video=payload["video"],
            )
        notification_type = {
            "performance": "VIDEO_RECOMMENDATION",
            "engagement": "SENTIMENT_ENGAGEMENT",
            "sentiment": "SENTIMENT_QA",
        }[flow]
        return self.generator.generate(
            prompt,
            user_id=payload.get("user_id"),
            notification_type=notification_type,
        )

    def _build_flow_context(
        self,
        user_id: int,
        flow: str,
    ) -> tuple[str, dict[str, Any]] | None:
        profile = self._get_user_profile(
            user_id,
            require_video_language=flow == "performance",
        )
        if profile is None:
            return None
        user_name = str(profile["user_name"]).strip() or f"User {user_id}"
        language = str(profile["app_language_code"]).strip()
        if flow == "engagement":
            response_data = get_user_response_rate(user_id)
            if not response_data or response_data["questions_sent"] == 0:
                self.last_skip_reason = NO_ENGAGEMENT_DATA
                return None
            return user_name, {
                "user_id": user_id,
                "language": language,
                "response_data": response_data,
                "context": "your CLAN participation",
            }
        if flow == "sentiment":
            eligible_responses = get_eligible_user_qa(user_id, self.db_engine)
            if not eligible_responses:
                self.last_skip_reason = NO_QA_DATA
                return user_name, {
                    "language": language,
                    "prepared_qa": None,
                    "history_records": [],
                    "context": "your recent responses and workplace reflections",
                }
            selection_type = eligible_responses[0].get("selection_type", "NEW")
            for response in eligible_responses:
                logger.info(
                    "Q&A selected user_id=%s question_id=%s answer_id=%s type=%s",
                    response["user_id"],
                    response["question_id"],
                    response["answer_id"],
                    selection_type,
                )
            prepared_qa = prepare_user_qa(
                user_id,
                self.db_engine,
                rows=eligible_responses,
            )
            return user_name, {
                "user_id": user_id,
                "language": language,
                "prepared_qa": prepared_qa,
                "history_records": eligible_responses,
                "context": "your recent responses and workplace reflections",
            }

        # Performance calculation and video retrieval remain separate concerns.
        language_id = profile["video_language_ids"]
        if not language_id:
            telugu_language_id = self._get_telugu_video_language_id()
            if telugu_language_id is None:
                self.last_skip_reason = MISSING_REQUIRED_DATA
                return None
            language_id = [telugu_language_id]
        calc = calculate_performance(user_id)
        weakest = calc.get("improvement_area")
        if not weakest:
            self.last_skip_reason = MISSING_REQUIRED_DATA
            return None
        performance = type(
            "PerformanceContext",
            (),
            {"kii_name": weakest["kii_name"], "kii_id": weakest["kii_id"]},
        )()
        recommendation = recommend_video(
            performance,
            language_id,
            embed_text,
            self.db_engine,
            user_id=user_id,
        )
        if recommendation is None:
            self.last_skip_reason = NO_VIDEO_RECOMMENDATION
            return None

        return user_name, {
            "user_id": user_id,
            "language": language,
            "weakest_kii": weakest,
            "video": recommendation,
            "video_title": recommendation.get("title"),
            "video_id": recommendation.get("video_id"),
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
        eligible_responses = get_eligible_user_qa(user_id, self.db_engine)
        if not eligible_responses:
            return {"user_id": user_id, "responses": []}

        selected_response = eligible_responses[0]
        selection_type = selected_response.get("selection_type", "NEW")
        logger.info(
            "Q&A selected user_id=%s question_id=%s answer_id=%s type=%s",
            selected_response["user_id"],
            selected_response["question_id"],
            selected_response["answer_id"],
            selection_type,
        )

        return {
            "user_id": user_id,
            "responses": [{
                "question_id": selected_response["question_id"],
                "question": selected_response["question"],
                "answer_id": selected_response["answer_id"],
                "answer": selected_response["answer"],
            }],
        }

    def build_notification(
        self,
        user_id: NotificationRequest | int,
        *,
        flow: str | None = None,
        should_send: bool | None = None,
    ) -> NotificationProcessingResult | None:
        self.last_skip_reason = None
        if isinstance(user_id, NotificationRequest):
            request = user_id
            user_id = request.user_id
            flow = flow or request.flow
            should_send = request.should_send if should_send is None else should_send
        else:
            should_send = True if should_send is None else should_send

        if user_id <= 0:
            raise ValueError("user_id must be greater than zero")

        if flow is None:
            event_type = get_next_manual_notification_for_user(
                user_id,
                db_engine=self.db_engine,
            )
            flow = FLOW_BY_EVENT_TYPE[event_type]

        notification_type = {
            "performance": "VIDEO_RECOMMENDATION",
            "engagement": "SENTIMENT_ENGAGEMENT",
            "sentiment": "SENTIMENT_QA",
        }[flow]

        flow_context = self._build_flow_context(user_id, flow)
        if flow_context is None:
            return None
        user_name, payload = flow_context
        if flow == "sentiment" and not payload["history_records"]:
            self.last_skip_reason = NO_QA_DATA
            return None
        try:
            notification_data = self._generate_llm_notification(flow, user_name, payload)
        except Exception:
            self.last_skip_reason = LLM_NO_RESPONSE
            logger.exception("LLM returned no usable notification for user=%s flow=%s", user_id, flow)
            return None
        if (
            not isinstance(notification_data, dict)
            or not isinstance(notification_data.get("title"), str)
            or not notification_data["title"].strip()
            or not isinstance(notification_data.get("description"), str)
            or not notification_data["description"].strip()
        ):
            self.last_skip_reason = LLM_NO_RESPONSE
            return None

        if flow == "performance":
            video = payload["video"]
            reference_id = int(video["video_id"])
        else:
            reference_id = 0
        video_popup = True if flow == "performance" else None

        notification = self.engine.make_notification(
            user_id=user_id,
            notification_type=notification_type,
            title=notification_data["title"],
            description=notification_data["description"],
            reference_id=reference_id,
            video_popup=video_popup,
        )

        response = NotificationProcessingResult(
            user_id=notification.user_id,
            title=notification.title,
            description=notification.description,
            notification_type=notification.notification_type,
            reference_id=notification.reference_id,
            video_popup=notification.video_popup,
            image=notification.image,
            flow=flow,
            should_send=should_send,
        )

        if should_send:
            if not self.sender.remote_url:
                response.remote_send_status = "skipped"
                response.error = "REMOTE_NOTIFICATION_SEND_URL is not configured"
            else:
                try:
                    remote_response = self.sender.send(
                        user_id=notification.user_id,
                        notification_type=notification.notification_type,
                        title=notification.title,
                        description=notification.description,
                        reference_id=notification.reference_id,
                        video_popup=notification.video_popup,
                        image=notification.image,
                    )
                    response.remote_send_status = "sent"
                    response.remote_send_response = remote_response
                    if flow == "sentiment":
                        save_sentiment_notification_history(
                            payload["history_records"],
                            self.db_engine,
                        )
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.exception(
                        "Notification send failed for user=%s flow=%s",
                        user_id,
                        flow,
                    )
                    response.remote_send_status = "failed"
                    response.error = str(exc)

        return response


__all__ = ["NotificationService"]
