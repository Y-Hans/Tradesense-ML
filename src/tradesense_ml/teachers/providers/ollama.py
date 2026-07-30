"""Ollama Teacher provider interface stub."""

from typing import Any

from tradesense_ml.domain.schemas.teacher import TeacherRequest
from tradesense_ml.teachers.base import BaseTeacherProvider


class OllamaTeacherProvider(BaseTeacherProvider):
    """Ollama local server provider implementation interface."""

    def __init__(self, default_model: str = "llama3:8b") -> None:
        super().__init__(
            provider_name="ollama",
            default_model=default_model,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
        )

    def _do_generate(self, request: TeacherRequest) -> tuple[str, dict[str, Any] | None, int, int]:
        mock_response = f"[Ollama:{self.default_model}] Ollama inference for {request.request_id}."
        return mock_response, None, 140, 220
