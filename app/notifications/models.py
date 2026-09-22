from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator


FlowName = Literal["performance", "engagement", "sentiment"]


class NotificationSendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int


class NotificationRequest(BaseModel):
    user_id: int
    flow: FlowName | None = None
    should_send: bool = True


class Notification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int
    title: str
    description: str | None = None
    notification_type: str
    reference_id: int = 0
    video_popup: bool | None = None
    image: str | None = None


class NotificationProcessingResult(BaseModel):
    user_id: int
    title: str
    description: str | None = None
    notification_type: str
    reference_id: int = 0
    video_popup: bool | None = None
    image: str | None = None
    flow: FlowName
    should_send: bool = True

    error: Optional[str] = None
    remote_send_status: Optional[str] = None
    remote_send_response: Optional[dict] = None


NotificationResponse = Notification


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
