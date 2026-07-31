"""Concrete Teacher Inference Pipeline orchestrating CoachRequest to validated CoachResponse conversion."""

import time
from datetime import UTC, datetime
from typing import Any

from tradesense_ml.assets_manager.manager import AssetManager
from tradesense_ml.domain.schemas.coaching import CoachRequest, CoachResponse
from tradesense_ml.domain.schemas.teacher import TeacherResponse
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.pipelines.inference.base import BaseInferencePipeline, BaseInferenceStrategy
from tradesense_ml.pipelines.inference.strategies import SingleTeacherStrategy
from tradesense_ml.teachers.prompt_builder import PromptBuilder
from tradesense_ml.teachers.prompt_renderer import PromptRenderer
from tradesense_ml.teachers.providers.anthropic import AnthropicTeacherProvider
from tradesense_ml.teachers.providers.gemini import GeminiTeacherProvider
from tradesense_ml.teachers.providers.local import LocalTeacherProvider
from tradesense_ml.teachers.providers.ollama import OllamaTeacherProvider
from tradesense_ml.teachers.providers.openai import OpenAITeacherProvider
from tradesense_ml.teachers.providers.openrouter import OpenRouterTeacherProvider
from tradesense_ml.teachers.providers.vllm import VLLMTeacherProvider
from tradesense_ml.teachers.response_parser import ResponseParser
from tradesense_ml.teachers.retry import RetryConfig, RetryHandler
from tradesense_ml.teachers.router import TeacherRouter
from tradesense_ml.teachers.validator import ResponseValidator

logger = get_logger()


class TeacherInferencePipeline(BaseInferencePipeline):
    """Concrete orchestrator for converting CoachRequest domain models into validated CoachResponse payloads."""

    def __init__(
        self,
        router: TeacherRouter | None = None,
        asset_manager: AssetManager | None = None,
        prompt_builder: PromptBuilder | None = None,
        prompt_renderer: PromptRenderer | None = None,
        retry_handler: RetryHandler | None = None,
        strategy: BaseInferenceStrategy | None = None,
    ) -> None:
        super().__init__(pipeline_name="teacher_inference_pipeline")

        # Initialize default providers if router not supplied
        if router is None:
            providers = [
                OpenRouterTeacherProvider(),
                OpenAITeacherProvider(),
                AnthropicTeacherProvider(),
                GeminiTeacherProvider(),
                OllamaTeacherProvider(),
                LocalTeacherProvider(),
                VLLMTeacherProvider(),
            ]
            self.router = TeacherRouter(providers)
        else:
            self.router = router

        self.asset_manager = asset_manager or AssetManager()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.prompt_renderer = prompt_renderer or PromptRenderer(asset_manager=self.asset_manager)
        self.retry_handler = retry_handler or RetryHandler(RetryConfig())
        self.strategy = strategy or SingleTeacherStrategy()

    def run(self, input_data: CoachRequest, **kwargs: Any) -> CoachResponse:
        """Run complete teacher inference pipeline synchronously."""
        start_time = time.perf_counter()
        logger.info(f"Starting teacher inference for CoachRequest '{input_data.request_id}'")

        # Configuration options
        prompt_version = kwargs.get("prompt_version", "v1")
        target_provider = kwargs.get("provider", kwargs.get("target_provider"))
        model_override = kwargs.get("model_name", kwargs.get("model"))
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 2048)

        # 1. Build prompt context
        prompt_context = self.prompt_builder.build_context(
            request=input_data, prompt_version=prompt_version
        )

        # 2. Render prompt into provider-independent RenderedPrompt
        rendered_prompt = self.prompt_renderer.render(context=prompt_context)

        # 3. Execution function with retry support
        def _execute_inference() -> tuple[TeacherResponse, CoachResponse]:
            # Temporarily set model override on target provider if provided
            selected_provider_key = target_provider or list(self.router.providers.keys())[0]
            prov = self.router.providers.get(selected_provider_key)
            original_model = None
            if prov and model_override:
                original_model = prov.default_model
                prov.default_model = model_override

            try:
                raw_response = self.strategy.execute(
                    request=input_data,
                    rendered_prompt=rendered_prompt,
                    router=self.router,
                    target_provider=target_provider,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # If multi-teacher strategy returns a list, select primary response for single response parsing
                teacher_resp: TeacherResponse = (
                    raw_response[0] if isinstance(raw_response, list) else raw_response
                )

                # Parse provider output into CoachResponse schema
                parse_target = (
                    teacher_resp.parsed_json
                    if teacher_resp.parsed_json is not None
                    else teacher_resp.content
                )
                parsed_coach_resp = ResponseParser.parse(
                    raw_output=parse_target, request_id=input_data.request_id
                )

                # Validate response against domain business constraints
                ResponseValidator.validate_and_raise(parsed_coach_resp)

                return teacher_resp, parsed_coach_resp
            finally:
                if prov and original_model is not None:
                    prov.default_model = original_model

        # Execute via RetryHandler
        teacher_response, coach_response = self.retry_handler.execute(_execute_inference)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Build inference metadata dictionary
        inference_metadata: dict[str, Any] = {
            "provider": teacher_response.provider_metadata.provider_name,
            "model": teacher_response.provider_metadata.model_name,
            "prompt_version": prompt_version,
            "latency_ms": round(total_latency_ms, 2),
            "token_usage": teacher_response.usage.model_dump(),
            "finish_reason": teacher_response.provider_metadata.finish_reason,
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": input_data.request_id,
            "strategy": type(self.strategy).__name__,
        }

        # Merge existing response metadata
        updated_metadata = {**coach_response.metadata, **inference_metadata}

        # Return updated CoachResponse with metadata
        return coach_response.model_copy(update={"metadata": updated_metadata})
