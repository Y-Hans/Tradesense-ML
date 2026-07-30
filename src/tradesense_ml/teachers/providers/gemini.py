"""Gemini Teacher provider interface stub."""

from typing import Any

from tradesense_ml.domain.schemas.teacher import TeacherRequest
from tradesense_ml.teachers.base import BaseTeacherProvider


class GeminiTeacherProvider(BaseTeacherProvider):
    """Google Gemini provider implementation interface."""

    def __init__(self, default_model: str = "gemini-1.5-pro") -> None:
        super().__init__(
            provider_name="gemini",
            default_model=default_model,
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.005,
        )

    def _do_generate(self, request: TeacherRequest) -> tuple[str, dict[str, Any] | None, int, int]:
        mock_response = f"[Gemini:{self.default_model}] Response for {request.request_id}."
        return mock_response, None, 130, 260
