"""Teacher inference execution strategies supporting single-teacher and multi-teacher orchestration."""

from typing import Any

from tradesense_ml.domain.schemas.coaching import CoachRequest
from tradesense_ml.domain.schemas.teacher import RenderedPrompt, TeacherResponse
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.pipelines.inference.base import BaseInferenceStrategy
from tradesense_ml.teachers.router import TeacherRouter

logger = get_logger()


class SingleTeacherStrategy(BaseInferenceStrategy):
    """Single teacher model inference strategy."""

    def __init__(self, provider_name: str | None = None) -> None:
        self.provider_name = provider_name

    def execute(
        self,
        request: CoachRequest,
        rendered_prompt: RenderedPrompt,
        router: TeacherRouter,
        **kwargs: Any,
    ) -> TeacherResponse:
        """Route rendered prompt to a single teacher model provider."""
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 2048)
        target_provider = kwargs.get("target_provider", self.provider_name)

        teacher_req = rendered_prompt.to_teacher_request(
            request_id=request.request_id,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=kwargs.get("extra_params"),
        )

        return router.route(teacher_req, target_provider=target_provider)


class MultiTeacherStrategy(BaseInferenceStrategy):
    """Multi-teacher parallel inference strategy (extensibility hook for multi-model orchestration)."""

    def __init__(self, provider_names: list[str] | None = None) -> None:
        self.provider_names = provider_names

    def execute(
        self,
        request: CoachRequest,
        rendered_prompt: RenderedPrompt,
        router: TeacherRouter,
        **kwargs: Any,
    ) -> list[TeacherResponse]:
        """Execute request across multiple teacher providers."""
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 2048)
        targets = kwargs.get("provider_names", self.provider_names)

        teacher_req = rendered_prompt.to_teacher_request(
            request_id=request.request_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return router.consensus_generate(teacher_req, provider_names=targets)


class ConsensusStrategy(BaseInferenceStrategy):
    """Consensus aggregation strategy hook for combining multiple teacher outputs."""

    def __init__(self, provider_names: list[str] | None = None) -> None:
        self.provider_names = provider_names

    def execute(
        self,
        request: CoachRequest,
        rendered_prompt: RenderedPrompt,
        router: TeacherRouter,
        **kwargs: Any,
    ) -> list[TeacherResponse]:
        """Gather responses from multiple teachers for consensus resolution."""
        multi_strat = MultiTeacherStrategy(provider_names=self.provider_names)
        return multi_strat.execute(request, rendered_prompt, router, **kwargs)
