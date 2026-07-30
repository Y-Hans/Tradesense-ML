"""Schemas for dataset examples: reviewed examples, training examples, and evaluation benchmark examples."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.coaching import CoachRequest, CoachResponse
from tradesense_ml.domain.schemas.review import ReviewAuditRecord, ReviewStage


class ReviewedExample(BaseModel):
    """Dataset example that has progressed through review pipeline."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., description="Unique dataset example ID")
    request: CoachRequest = Field(..., description="Original input coach request")
    teacher_response: CoachResponse = Field(..., description="Generated coaching response")
    review_status: ReviewStage = Field(..., description="Current review status")
    audit_trail: list[ReviewAuditRecord] = Field(
        default_factory=list, description="Complete audit history of reviews"
    )
    final_quality_score: float | None = Field(
        default=None, ge=0.0, le=10.0, description="Final approved quality score"
    )


class TrainingMessage(BaseModel):
    """Single chat message format for chat fine-tuning formats (e.g. OpenAI/HuggingFace)."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message text content")


class TrainingExample(BaseModel):
    """Fine-tuning formatted dataset record."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., description="Source example ID")
    messages: list[TrainingMessage] = Field(..., description="Structured conversational messages")
    prompt_text: str = Field(..., description="Formatted input prompt text")
    target_text: str = Field(..., description="Expected model target response text")
    dataset_version: str = Field(..., description="Dataset version reference")
    weight: float = Field(default=1.0, ge=0.0, description="Sample weight in loss function")


class EvaluationExample(BaseModel):
    """Benchmark evaluation sample container."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(..., description="Benchmark sample ID")
    benchmark_id: str = Field(..., description="Associated benchmark suite ID")
    input_request: CoachRequest = Field(..., description="Input query request")
    ground_truth_response: CoachResponse | None = Field(
        default=None, description="Gold-standard ground truth response if available"
    )
    expected_reason_codes: list[str] = Field(
        default_factory=list, description="List of reason codes expected to be identified"
    )
    evaluation_criteria: list[str] = Field(
        default_factory=list, description="Criteria names to score"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Sample metadata")
