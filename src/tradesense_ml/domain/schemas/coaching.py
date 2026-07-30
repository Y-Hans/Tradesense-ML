"""Coaching request and response schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.evaluations import DisciplineEvaluation, RiskEvaluation
from tradesense_ml.domain.schemas.market_context import MarketContext
from tradesense_ml.domain.schemas.trade import Trade


class CoachRequest(BaseModel):
    """User coaching request payload."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(..., description="Unique coaching request ID")
    user_id: str = Field(..., description="User ID requesting coaching")
    trade: Trade = Field(..., description="Associated trade object")
    market_context: MarketContext | None = Field(
        default=None, description="Optional market context"
    )
    user_notes: str | None = Field(
        default=None, description="User comments/rationale for the trade"
    )
    requested_aspects: list[str] = Field(
        default_factory=lambda: ["risk", "discipline", "general"],
        description="Aspects to evaluate",
    )


class CoachResponse(BaseModel):
    """AI Coaching response payload delivered to user."""

    model_config = ConfigDict(frozen=True)

    response_id: str = Field(..., description="Unique response ID")
    request_id: str = Field(..., description="Reference to request ID")
    headline: str = Field(..., description="Single sentence summary headline")
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Combined coaching score")
    risk_evaluation: RiskEvaluation = Field(..., description="Detailed risk evaluation")
    discipline_evaluation: DisciplineEvaluation = Field(
        ..., description="Detailed discipline evaluation"
    )
    actionable_advice: list[str] = Field(
        ..., description="Bulleted list of concrete actionable steps"
    )
    educational_note: str = Field(
        ..., description="Educational explanation of trading concepts involved"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution and model metadata"
    )
