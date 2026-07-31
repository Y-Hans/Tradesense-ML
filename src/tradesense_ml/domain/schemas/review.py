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


class ReviewVerdict(str, Enum):
    """Review verdict decision."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    NEEDS_REVISION = "NEEDS_REVISION"
    ESCALATE = "ESCALATE"


# Alias for backward compatibility where ReviewDecision was used as enum
ReviewDecisionEnum = ReviewVerdict


class ReasonCode(str, Enum):
    """Structured reason codes for review evaluation results."""

    GOOD_RISK_ANALYSIS = "GOOD_RISK_ANALYSIS"
    GOOD_DISCIPLINE_ANALYSIS = "GOOD_DISCIPLINE_ANALYSIS"
    GOOD_ACTION_PLAN = "GOOD_ACTION_PLAN"
    INCONSISTENT_REASONING = "INCONSISTENT_REASONING"
    MISSING_ACTIONABLE_ADVICE = "MISSING_ACTIONABLE_ADVICE"
    LOW_EDUCATIONAL_VALUE = "LOW_EDUCATIONAL_VALUE"
    HALLUCINATED_MARKET_FACT = "HALLUCINATED_MARKET_FACT"
    INSUFFICIENT_EXPLANATION = "INSUFFICIENT_EXPLANATION"
    UNSAFE_CONTENT = "UNSAFE_CONTENT"
    STYLE_VIOLATION = "STYLE_VIOLATION"
    INCOMPLETE_RESPONSE = "INCOMPLETE_RESPONSE"
    EXCELLENT_COACHING = "EXCELLENT_COACHING"


class ReviewResult(BaseModel):
    """Intermediate raw evaluation payload produced by a reviewer prior to decision policy resolution."""

    model_config = ConfigDict(frozen=True)

    evaluation_id: str = Field(..., description="Unique evaluation result ID")
    response_id: str = Field(..., description="Target CoachResponse ID")
    quality_score: float = Field(..., ge=0.0, le=10.0, description="Raw quality score (0-10)")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Reviewer confidence score (0-1)"
    )
    reviewer_name: str = Field(..., description="Name of the reviewer")
    reviewer_type: str = Field(..., description="Type of reviewer")
    passed_checks: list[str] = Field(
        default_factory=list, description="Passed quality criteria checks"
    )
    failed_checks: list[str] = Field(
        default_factory=list, description="Failed quality criteria checks"
    )
    reason_codes: list[ReasonCode] = Field(
        default_factory=list, description="Structured reason codes"
    )
    revision_suggestions: list[str] = Field(
        default_factory=list, description="Raw revision suggestions"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Raw reviewer metadata and score breakdown"
    )


class ReviewDecision(BaseModel):
    """Structured standard review output object delivered by the Review Pipeline."""

    model_config = ConfigDict(frozen=True)

    review_id: str = Field(..., description="Unique review decision ID")
    response_id: str = Field(..., description="Target CoachResponse ID")
    verdict: ReviewVerdict = Field(..., description="Review verdict decision")
    quality_score: float = Field(..., ge=0.0, le=10.0, description="Overall quality score (0-10)")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Reviewer confidence score (0-1)"
    )
    reviewer_name: str = Field(..., description="Name of the reviewer")
    reviewer_type: str = Field(..., description="Type of reviewer")
    review_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of review execution"
    )
    review_duration_ms: float = Field(
        default=0.0, ge=0.0, description="Execution duration in milliseconds"
    )
    passed_checks: list[str] = Field(
        default_factory=list, description="Passed quality criteria checks"
    )
    failed_checks: list[str] = Field(
        default_factory=list, description="Failed quality criteria checks"
    )
    reason_codes: list[ReasonCode] = Field(
        default_factory=list, description="Structured reason codes"
    )
    revision_suggestions: list[str] = Field(
        default_factory=list, description="Actionable revision suggestions"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Review execution and config metadata"
    )


class ReviewAuditRecord(BaseModel):
    """Immutable audit trail record for a review action."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(..., description="Unique audit record ID")
    stage: ReviewStage = Field(..., description="Review stage during which action occurred")
    reviewer_id: str = Field(
        ...,
        description="Reviewer identifier (e.g. system_validator, teacher_gpt4, human_annotator_42)",
    )
    decision: ReviewVerdict = Field(..., description="Verdict rendered")
    score: float | None = Field(default=None, ge=0.0, le=10.0, description="Quality score (0-10)")
    comments: str | None = Field(default=None, description="Reviewer notes or feedback")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of review")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional review context")
