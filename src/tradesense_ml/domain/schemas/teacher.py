"""Teacher model schemas for request, response, token, and cost tracking."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    """Token consumption and estimated financial cost."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0, description="Input prompt tokens")
    completion_tokens: int = Field(default=0, ge=0, description="Output completion tokens")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed")
    estimated_cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated cost in USD")


class ProviderMetadata(BaseModel):
    """Metadata regarding teacher LLM provider execution."""

    model_config = ConfigDict(frozen=True)

    provider_name: str = Field(
        ..., description="Provider identifier (e.g. openrouter, openai, local)"
    )
    model_name: str = Field(..., description="Exact provider model ID")
    latency_ms: float = Field(..., ge=0.0, description="Inference duration in milliseconds")
    finish_reason: str | None = Field(default="stop", description="Generation completion status")
    raw_response_id: str | None = Field(default=None, description="Native API response ID")


class TeacherRequest(BaseModel):
    """Standardized request sent to a Teacher model."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(..., description="Unique teacher request ID")
    system_prompt: str = Field(..., description="System instructions")
    user_prompt: str = Field(..., description="User query / prompt payload")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Max response tokens")
    prompt_version: str = Field(default="v1", description="Prompt version identifier")
    extra_params: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific settings"
    )


class RenderedPrompt(BaseModel):
    """Provider-independent rendered prompt container containing system and user prompts."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str = Field(..., description="Rendered system prompt instructions")
    user_prompt: str = Field(..., description="Rendered user prompt payload")
    prompt_version: str = Field(default="v1", description="Prompt version identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Rendering metadata and variables"
    )

    def to_teacher_request(
        self,
        request_id: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        extra_params: dict[str, Any] | None = None,
    ) -> TeacherRequest:
        """Adapt provider-independent RenderedPrompt into a TeacherRequest."""
        return TeacherRequest(
            request_id=request_id,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_version=self.prompt_version,
            extra_params=extra_params or {},
        )


class TeacherResponse(BaseModel):
    """Standardized response received from a Teacher model."""

    model_config = ConfigDict(frozen=True)

    response_id: str = Field(..., description="Unique teacher response ID")
    request_id: str = Field(..., description="Associated request ID")
    content: str = Field(..., description="Raw text response content")
    parsed_json: dict[str, Any] | None = Field(
        default=None, description="Parsed JSON object if structured output requested"
    )
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Token usage details")
    provider_metadata: ProviderMetadata = Field(..., description="Provider metadata")
