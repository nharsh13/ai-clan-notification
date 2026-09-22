import pytest
from pydantic import ValidationError

from app.notifications.engine import NotificationEngine
from app.notifications.models import Notification


def test_notification_model_matches_public_contract():
	notification = Notification(
		user_id=953,
		title="Hello veera",
		description="Stay curious.",
		notification_type="SENTIMENT_QA",
	)

	assert notification.model_dump() == {
		"user_id": 953,
		"title": "Hello veera",
		"description": "Stay curious.",
		"notification_type": "SENTIMENT_QA",
		"reference_id": 0,
		"video_popup": None,
		"image": None,
	}


def test_notification_model_rejects_legacy_fields():
	with pytest.raises(ValidationError):
		Notification(
			user_id=953,
			title="Title",
			notification_type="SENTIMENT_QA",
			notification_title="Legacy title",
		)


def test_notification_engine_make_notification():
	notification = NotificationEngine().make_notification(
		user_id=953,
		title="Title",
		description=None,
		notification_type="VIDEO_RECOMMENDATION",
		reference_id=55,
		video_popup=True,
		image="/image.png",
	)

	assert notification.model_dump() == {
		"user_id": 953,
		"title": "Title",
		"description": None,
		"notification_type": "VIDEO_RECOMMENDATION",
		"reference_id": 55,
		"video_popup": True,
		"image": "/image.png",
	}
