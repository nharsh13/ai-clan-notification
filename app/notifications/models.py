from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


CampaignDay = Literal[
    22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
]

FlowName = Literal["performance", "engagement", "sentiment"]


class NotificationRequest(BaseModel):
    user_id: int
    flow: FlowName = "performance"
    weak_indicator: Optional[int] = None
    notification_type: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    reference_id: Optional[int] = None
    video_id: Optional[int] = None
    video_title: Optional[str] = None
    creator_name: Optional[str] = None
    deep_link: Optional[str] = None
    should_send: bool = True
    video_popup: Optional[str] = None
    campaign_day: Optional[CampaignDay] = None


class Notification(BaseModel):
    user_id: int
    flow: FlowName = "performance"
    campaign_day: Optional[CampaignDay] = None

    notification_title: str = Field(max_length=140)
    notification_body: str = Field(max_length=500)

    audience_strategy: Optional[str] = None
    cohort_key: Optional[str] = None
    action: Optional[str] = None
    deep_link: Optional[str] = None

    notification_type: Optional[str] = None
    should_send: bool = True

    reference_id: Optional[int] = None
    video_id: Optional[int] = None
    video_title: Optional[str] = None
    creator_name: Optional[str] = None
    video_popup: Optional[str] = None


class NotificationResponse(BaseModel):
    user_id: int
    flow: FlowName = "performance"
    campaign_day: Optional[CampaignDay] = None
    notification_title: str = Field(max_length=140)
    notification_body: str = Field(max_length=500)

    audience_strategy: Optional[str] = None
    cohort_key: Optional[str] = None
    action: Optional[str] = None
    deep_link: Optional[str] = None

    notification_type: Optional[str] = None
    should_send: bool = True

    reference_id: Optional[int] = None
    video_id: Optional[int] = None
    video_title: Optional[str] = None
    creator_name: Optional[str] = None
    video_popup: Optional[str] = None

    error: Optional[str] = None
    remote_send_status: Optional[str] = None
    remote_send_response: Optional[dict] = None

    @field_validator("notification_title")
    @classmethod
    def validate_title(cls, v: str):
        text = str(v).strip()

        if not text:
            raise ValueError("notification_title cannot be empty")

        return text

    @field_validator("notification_body")
    @classmethod
    def validate_body(cls, v: str):
        text = str(v).strip()

        if not text:
            raise ValueError("notification_body cannot be empty")

        return text


class BatchNotificationRequest(BaseModel):
    items: list[NotificationRequest]

    @field_validator("items")
    @classmethod
    def validate_items(cls, v: list[NotificationRequest]):
        if not v:
            raise ValueError("items cannot be empty")

        if len(v) > 5000:
            raise ValueError("items cannot exceed 5000 in one call")

        return v


class BatchNotificationResponse(BaseModel):
    total: int
    results: list[NotificationResponse]


class SendNotificationResponse(BaseModel):
    success: bool
    user_id: int
    notification: NotificationResponse

    error: Optional[str] = None
    remote_send_status: Optional[str] = None
    remote_send_response: Optional[dict] = None