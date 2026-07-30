"""OpenRouter Teacher provider interface stub."""

from typing import Any

from tradesense_ml.domain.schemas.teacher import TeacherRequest
from tradesense_ml.teachers.base import BaseTeacherProvider


class OpenRouterTeacherProvider(BaseTeacherProvider):
    """OpenRouter provider implementation interface."""

    def __init__(self, default_model: str = "anthropic/claude-3.5-sonnet") -> None:
        super().__init__(
            provider_name="openrouter",
            default_model=default_model,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
        )

    def _do_generate(self, request: TeacherRequest) -> tuple[str, dict[str, Any] | None, int, int]:
        # Interface stub - future API call via OpenRouter REST endpoint
        mock_response = f"[OpenRouter:{self.default_model}] Coaching response generated for request {request.request_id}."
        return mock_response, None, 150, 300
