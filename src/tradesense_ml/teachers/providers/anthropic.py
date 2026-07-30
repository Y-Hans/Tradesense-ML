"""Anthropic Teacher provider interface stub."""

from typing import Any

from tradesense_ml.domain.schemas.teacher import TeacherRequest
from tradesense_ml.teachers.base import BaseTeacherProvider


class AnthropicTeacherProvider(BaseTeacherProvider):
    """Anthropic provider implementation interface."""

    def __init__(self, default_model: str = "claude-3-5-sonnet-20241022") -> None:
        super().__init__(
            provider_name="anthropic",
            default_model=default_model,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
        )

    def _do_generate(self, request: TeacherRequest) -> tuple[str, dict[str, Any] | None, int, int]:
        mock_response = f"[Anthropic:{self.default_model}] Response for {request.request_id}."
        return mock_response, None, 160, 320
