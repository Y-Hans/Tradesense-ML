"""vLLM Teacher provider interface stub."""

from typing import Any

from tradesense_ml.domain.schemas.teacher import TeacherRequest
from tradesense_ml.teachers.base import BaseTeacherProvider


class VLLMTeacherProvider(BaseTeacherProvider):
    """vLLM high-throughput local engine provider interface."""

    def __init__(self, default_model: str = "mistralai/Mistral-7B-Instruct-v0.2") -> None:
        super().__init__(
            provider_name="vllm",
            default_model=default_model,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
        )

    def _do_generate(self, request: TeacherRequest) -> tuple[str, dict[str, Any] | None, int, int]:
        mock_response = (
            f"[vLLM:{self.default_model}] vLLM inference response for {request.request_id}."
        )
        return mock_response, None, 160, 240
