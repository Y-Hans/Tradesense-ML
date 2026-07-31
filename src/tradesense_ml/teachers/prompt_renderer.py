"""Prompt Renderer module for loading asset templates and producing provider-independent RenderedPrompt objects."""

from typing import Any

from tradesense_ml.assets_manager.manager import AssetManager
from tradesense_ml.domain.schemas.teacher import RenderedPrompt
from tradesense_ml.teachers.prompt_builder import PromptContext


class PromptRenderer:
    """Renders versioned system and user prompt templates into provider-independent RenderedPrompt objects."""

    def __init__(self, asset_manager: AssetManager | None = None) -> None:
        """Initialize PromptRenderer with an AssetManager instance."""
        self.asset_manager = asset_manager or AssetManager()

    def render(
        self,
        context: PromptContext,
        system_template_name: str = "system_coach",
        user_template_name: str = "user_coach",
    ) -> RenderedPrompt:
        """Deterministically render system and user prompt templates from AssetManager."""
        version = context.prompt_version

        # 1. Load system prompt template
        system_template = self.asset_manager.get_prompt(system_template_name, version=version)
        system_prompt = system_template.strip()

        # 2. Load user prompt template
        user_template = self.asset_manager.get_prompt(user_template_name, version=version)

        # 3. Deterministic template variable interpolation
        rendered_user_prompt = user_template.format(
            trade_record=context.trade_record_text,
            market_context=context.market_context_text,
            user_notes=context.user_notes_text,
            requested_aspects=context.requested_aspects_text,
        ).strip()

        metadata: dict[str, Any] = {
            "request_id": context.request_id,
            "user_id": context.user_id,
            "prompt_version": version,
            "system_template_name": system_template_name,
            "user_template_name": user_template_name,
            "extra_context": context.extra_context,
        }

        return RenderedPrompt(
            system_prompt=system_prompt,
            user_prompt=rendered_user_prompt,
            prompt_version=version,
            metadata=metadata,
        )
