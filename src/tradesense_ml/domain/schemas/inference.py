"""Inference serving request and response schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.coaching import CoachRequest, CoachResponse


class InferenceRequest(BaseModel):
    """API request wrapper for model inference."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(..., description="Unique client inference request ID")
    model_id: str | None = Field(default=None, description="Requested model ID or stage alias")
    coach_request: CoachRequest = Field(..., description="Trade coaching request payload")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Inference parameters (e.g. temperature, max_tokens)"
    )


class InferenceResponse(BaseModel):
    """API response wrapper for model inference."""

    model_config = ConfigDict(frozen=True)

    response_id: str = Field(..., description="Unique response ID")
    request_id: str = Field(..., description="Associated request ID")
    model_version: str = Field(..., description="Model version that served the request")
    coach_response: CoachResponse = Field(..., description="Generated coaching feedback")
    latency_ms: float = Field(..., ge=0.0, description="Inference duration in ms")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Serving system metadata")
