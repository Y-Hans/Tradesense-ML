"""OpenAI Teacher provider interface stub."""

from typing import Any

from tradesense_ml.domain.schemas.teacher import TeacherRequest
from tradesense_ml.teachers.base import BaseTeacherProvider


class OpenAITeacherProvider(BaseTeacherProvider):
    """OpenAI provider implementation interface."""

    def __init__(self, default_model: str = "gpt-4o") -> None:
        super().__init__(
            provider_name="openai",
            default_model=default_model,
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.010,
        )

    def _do_generate(self, request: TeacherRequest) -> tuple[str, dict[str, Any] | None, int, int]:
        mock_response = f"[OpenAI:{self.default_model}] Response for {request.request_id}."
        return mock_response, None, 140, 280
