"""Evaluation rubric schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RubricCriterion(BaseModel):
    """Single criterion definition in an evaluation rubric."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Short identifier of criterion")
    description: str = Field(..., description="Detailed description of what is evaluated")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Weighting factor (0.0 to 1.0)")
    max_score: float = Field(default=10.0, gt=0.0, description="Maximum possible score")


class Rubric(BaseModel):
    """Complete versioned evaluation rubric definition."""

    model_config = ConfigDict(frozen=True)

    rubric_id: str = Field(..., description="Unique rubric identifier")
    version: str = Field(..., description="Semantic version string, e.g. 1.0.0")
    title: str = Field(..., description="Human-readable title")
    criteria: list[RubricCriterion] = Field(..., description="List of criteria")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Rubric creation metadata")


class RubricScore(BaseModel):
    """Score awarded for a single criterion."""

    model_config = ConfigDict(frozen=True)

    criterion_name: str = Field(..., description="Name of evaluated criterion")
    score: float = Field(..., ge=0.0, description="Score awarded")
    max_score: float = Field(default=10.0, gt=0.0, description="Maximum score for criterion")
    reasoning: str = Field(..., description="Justification for the given score")


class EvaluationResult(BaseModel):
    """Result of evaluating a response against a rubric."""

    model_config = ConfigDict(frozen=True)

    evaluation_id: str = Field(..., description="Unique evaluation ID")
    rubric_id: str = Field(..., description="Evaluated rubric ID")
    rubric_version: str = Field(..., description="Evaluated rubric version")
    scores: list[RubricScore] = Field(..., description="Scores for each criterion")
    weighted_total_score: float = Field(
        ..., ge=0.0, le=10.0, description="Normalized weighted total score (0-10)"
    )
    passed: bool = Field(..., description="Whether evaluation met passing threshold")
    feedback_summary: str = Field(..., description="Qualitative evaluation feedback summary")
