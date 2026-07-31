"""Training strategies defining workflow execution patterns independently of backends."""

from abc import ABC, abstractmethod
from typing import ClassVar

from tradesense_ml.domain.schemas.distillation import DistillationArtifact
from tradesense_ml.domain.schemas.finetuning import TrainingBackendResult, TrainingConfiguration
from tradesense_ml.finetuning.backends.base import TrainingBackend


class TrainingStrategy(ABC):
    """Abstract top-level training strategy workflow interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy unique identifier."""

    @abstractmethod
    def execute(
        self,
        distillation_artifact: DistillationArtifact,
        backend: TrainingBackend,
        config: TrainingConfiguration,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
    ) -> TrainingBackendResult:
        """Execute the training workflow using the supplied backend.

        Args:
            distillation_artifact: Input DistillationArtifact.
            backend: Execution engine instance.
            config: Training configuration.
            output_dir: Output directory.
            resume_from_checkpoint: Optional checkpoint path to resume.

        Returns:
            TrainingBackendResult containing backend execution metadata and metrics.
        """


class SFTTrainingStrategy(TrainingStrategy):
    """Supervised Fine-Tuning (SFT) strategy for instruction tuning."""

    @property
    def name(self) -> str:
        return "SFTTrainingStrategy"

    def execute(
        self,
        distillation_artifact: DistillationArtifact,
        backend: TrainingBackend,
        config: TrainingConfiguration,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
    ) -> TrainingBackendResult:
        return backend.train(
            distillation_artifact=distillation_artifact,
            config=config,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )


class DPOTrainingStrategy(TrainingStrategy):
    """Direct Preference Optimization (DPO) strategy using preference pairs."""

    @property
    def name(self) -> str:
        return "DPOTrainingStrategy"

    def execute(
        self,
        distillation_artifact: DistillationArtifact,
        backend: TrainingBackend,
        config: TrainingConfiguration,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
    ) -> TrainingBackendResult:
        return backend.train(
            distillation_artifact=distillation_artifact,
            config=config,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )


class ORPOTrainingStrategy(TrainingStrategy):
    """Odds Ratio Preference Optimization (ORPO) monolithic strategy."""

    @property
    def name(self) -> str:
        return "ORPOTrainingStrategy"

    def execute(
        self,
        distillation_artifact: DistillationArtifact,
        backend: TrainingBackend,
        config: TrainingConfiguration,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
    ) -> TrainingBackendResult:
        return backend.train(
            distillation_artifact=distillation_artifact,
            config=config,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )


class CurriculumTrainingStrategy(TrainingStrategy):
    """Curriculum fine-tuning strategy progressing through difficulty stages."""

    @property
    def name(self) -> str:
        return "CurriculumTrainingStrategy"

    def execute(
        self,
        distillation_artifact: DistillationArtifact,
        backend: TrainingBackend,
        config: TrainingConfiguration,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
    ) -> TrainingBackendResult:
        return backend.train(
            distillation_artifact=distillation_artifact,
            config=config,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )


class HybridTrainingStrategy(TrainingStrategy):
    """Hybrid multi-stage strategy combining SFT and preference alignment."""

    @property
    def name(self) -> str:
        return "HybridTrainingStrategy"

    def execute(
        self,
        distillation_artifact: DistillationArtifact,
        backend: TrainingBackend,
        config: TrainingConfiguration,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
    ) -> TrainingBackendResult:
        return backend.train(
            distillation_artifact=distillation_artifact,
            config=config,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )


class TrainingStrategyRegistry:
    """Registry pattern for discovering and instantiating training strategies."""

    _registry: ClassVar[dict[str, type[TrainingStrategy]]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: type[TrainingStrategy]) -> None:
        """Register a new TrainingStrategy implementation."""
        cls._registry[name.lower()] = strategy_cls

    @classmethod
    def get(cls, name: str) -> TrainingStrategy:
        """Retrieve an instantiated TrainingStrategy by registered name."""
        strategy_name = name.lower()
        if strategy_name not in cls._registry:
            available = ", ".join(cls.list_available())
            raise ValueError(
                f"Training strategy '{name}' is not registered. Available strategies: [{available}]"
            )
        return cls._registry[strategy_name]()

    @classmethod
    def list_available(cls) -> list[str]:
        """List names of all registered strategies."""
        return sorted(list(cls._registry.keys()))


# Register built-in strategies
TrainingStrategyRegistry.register("sfttrainingstrategy", SFTTrainingStrategy)
TrainingStrategyRegistry.register("sft", SFTTrainingStrategy)
TrainingStrategyRegistry.register("dpotrainingstrategy", DPOTrainingStrategy)
TrainingStrategyRegistry.register("dpo", DPOTrainingStrategy)
TrainingStrategyRegistry.register("orpotrainingstrategy", ORPOTrainingStrategy)
TrainingStrategyRegistry.register("orpo", ORPOTrainingStrategy)
TrainingStrategyRegistry.register("curriculumtrainingstrategy", CurriculumTrainingStrategy)
TrainingStrategyRegistry.register("curriculum", CurriculumTrainingStrategy)
TrainingStrategyRegistry.register("hybridtrainingstrategy", HybridTrainingStrategy)
TrainingStrategyRegistry.register("hybrid", HybridTrainingStrategy)
