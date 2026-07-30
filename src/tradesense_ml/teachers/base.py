"""Provider-agnostic Teacher model protocol and base class."""

import time
from abc import ABC, abstractmethod
from typing import Any

from tradesense_ml.domain.schemas.teacher import (
    ProviderMetadata,
    TeacherRequest,
    TeacherResponse,
    TokenUsage,
)
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class BaseTeacherProvider(ABC):
    """Abstract provider-agnostic interface for Teacher models."""

    def __init__(
        self,
        provider_name: str,
        default_model: str,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ) -> None:
        self.provider_name = provider_name
        self.default_model = default_model
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate estimated cost in USD based on token counts."""
        return (prompt_tokens / 1000.0) * self.cost_per_1k_input + (
            completion_tokens / 1000.0
        ) * self.cost_per_1k_output

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        """Synchronous generation entrypoint with latency and cost tracking."""
        start_time = time.perf_counter()
        logger.info(
            f"Sending request {request.request_id} to provider '{self.provider_name}' (model: {self.default_model})"
        )

        # Call concrete implementation stub
        content, parsed_json, prompt_tokens, completion_tokens = self._do_generate(request)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = self.estimate_cost(prompt_tokens, completion_tokens)

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost_usd,
        )

        metadata = ProviderMetadata(
            provider_name=self.provider_name,
            model_name=self.default_model,
            latency_ms=latency_ms,
            finish_reason="stop",
        )

        return TeacherResponse(
            response_id=f"resp_{request.request_id}",
            request_id=request.request_id,
            content=content,
            parsed_json=parsed_json,
            usage=usage,
            provider_metadata=metadata,
        )

    @abstractmethod
    def _do_generate(self, request: TeacherRequest) -> tuple[str, dict[str, Any] | None, int, int]:
        """Internal abstract method to execute API call.

        Returns tuple of (content_text, optional_parsed_json, prompt_tokens, completion_tokens).
        """
        pass
