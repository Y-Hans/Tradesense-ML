"""Review pipeline schemas for automated, AI, human, and approval lifecycle stages."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReviewStage(str, Enum):
    """Stages of dataset review lifecycle."""

    AUTOMATED_VALIDATION = "AUTOMATED_VALIDATION"
    AI_TEACHER_REVIEW = "AI_TEACHER_REVIEW"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ReviewDecision(str, Enum):
    """Review verdict decision."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    NEEDS_REVISION = "NEEDS_REVISION"
    ESCALATE = "ESCALATE"


class ReviewAuditRecord(BaseModel):
    """Immutable audit trail record for a review action."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(..., description="Unique audit record ID")
    stage: ReviewStage = Field(..., description="Review stage during which action occurred")
    reviewer_id: str = Field(
        ...,
        description="Reviewer identifier (e.g. system_validator, teacher_gpt4, human_annotator_42)",
    )
    decision: ReviewDecision = Field(..., description="Verdict rendered")
    score: float | None = Field(default=None, ge=0.0, le=10.0, description="Quality score (0-10)")
    comments: str | None = Field(default=None, description="Reviewer notes or feedback")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of review")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional review context")
