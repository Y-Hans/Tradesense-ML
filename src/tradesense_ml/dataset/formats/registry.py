"""Registry and factory for DatasetFormat abstractions."""

from tradesense_ml.dataset.formats.base import BaseDatasetFormat
from tradesense_ml.dataset.formats.evaluation import EvaluationFormat
from tradesense_ml.dataset.formats.preference import (
    DPOFormat,
    KTOFormat,
    ORPOFormat,
    RewardModelFormat,
)
from tradesense_ml.dataset.formats.sft import SFTChatFormat, SFTInstructionFormat
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class DatasetFormatRegistry:
    """Registry managing dataset format strategy implementations."""

    _FORMAT_INSTANCES: dict[str, BaseDatasetFormat] = {
        "sft_instruction": SFTInstructionFormat(),
        "sft_chat": SFTChatFormat(),
        "evaluation": EvaluationFormat(),
        "dpo": DPOFormat(),
        "orpo": ORPOFormat(),
        "kto": KTOFormat(),
        "reward_model": RewardModelFormat(),
    }

    @classmethod
    def get_format(cls, format_name: str) -> BaseDatasetFormat:
        """Retrieve a DatasetFormat strategy instance by name.

        Args:
            format_name: Name of requested format strategy.

        Returns:
            Instance of BaseDatasetFormat subclass.
        """
        key = format_name.lower().strip()
        return cls._FORMAT_INSTANCES.get(key, SFTInstructionFormat())

    @classmethod
    def register_format(cls, name: str, format_inst: BaseDatasetFormat) -> None:
        """Register a custom DatasetFormat strategy instance."""
        key = name.lower().strip()
        cls._FORMAT_INSTANCES[key] = format_inst
        logger.info(
            f"Registered custom DatasetFormat strategy: '{key}' -> {format_inst.__class__.__name__}"
        )

    @classmethod
    def list_formats(cls) -> list[str]:
        """List registered format strategy names."""
        return list(cls._FORMAT_INSTANCES.keys())
