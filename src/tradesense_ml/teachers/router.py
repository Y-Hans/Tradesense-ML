"""Teacher router supporting multi-teacher consensus and provider fallbacks."""

from tradesense_ml.domain.schemas.teacher import TeacherRequest, TeacherResponse
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.teachers.base import BaseTeacherProvider

logger = get_logger()


class TeacherRouter:
    """Routing engine for teacher model requests across multiple providers."""

    def __init__(self, providers: list[BaseTeacherProvider] | None = None) -> None:
        self.providers: dict[str, BaseTeacherProvider] = {}
        if providers:
            for p in providers:
                self.register_provider(p)

    def register_provider(self, provider: BaseTeacherProvider) -> None:
        """Register a teacher provider."""
        self.providers[provider.provider_name] = provider
        logger.info(f"Registered teacher provider: '{provider.provider_name}'")

    def route(self, request: TeacherRequest, target_provider: str | None = None) -> TeacherResponse:
        """Route request to specified or primary provider with fallback support."""
        if not self.providers:
            raise RuntimeError("No teacher providers registered in router.")

        provider_key = target_provider or list(self.providers.keys())[0]
        if provider_key not in self.providers:
            logger.warning(
                f"Provider '{provider_key}' not found. Falling back to '{list(self.providers.keys())[0]}'"
            )
            provider_key = list(self.providers.keys())[0]

        return self.providers[provider_key].generate(request)

    def consensus_generate(
        self, request: TeacherRequest, provider_names: list[str] | None = None
    ) -> list[TeacherResponse]:
        """Execute request across multiple teacher providers to gather consensus responses."""
        selected = provider_names or list(self.providers.keys())
        responses = []
        for name in selected:
            if name in self.providers:
                resp = self.providers[name].generate(request)
                responses.append(resp)
        return responses
