"""TrainingBackend abstract base class and backend registry."""

from abc import ABC, abstractmethod
from typing import ClassVar

from tradesense_ml.domain.schemas.distillation import DistillationArtifact
from tradesense_ml.domain.schemas.finetuning import TrainingBackendResult, TrainingConfiguration


class TrainingBackend(ABC):
    """Abstract interface for all model training backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique backend identifier string (e.g., mock, unsloth, axolotl, huggingface, trl)."""

    @abstractmethod
    def train(
        self,
        distillation_artifact: DistillationArtifact,
        config: TrainingConfiguration,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
    ) -> TrainingBackendResult:
        """Execute model training using the backend engine.

        Args:
            distillation_artifact: Canonical DistillationArtifact input containing training datasets.
            config: Training execution configuration.
            output_dir: Directory where weights, checkpoints, and logs should be saved.
            resume_from_checkpoint: Optional path to checkpoint to resume training from.

        Returns:
            Canonical TrainingBackendResult object containing metrics, weight directory, and status.
        """


class BackendRegistry:
    """Registry pattern for discovering and instantiating training backends."""

    _registry: ClassVar[dict[str, type[TrainingBackend]]] = {}

    @classmethod
    def register(cls, name: str, backend_cls: type[TrainingBackend]) -> None:
        """Register a new TrainingBackend implementation."""
        cls._registry[name.lower()] = backend_cls

    @classmethod
    def get(cls, name: str) -> TrainingBackend:
        """Retrieve an instantiated TrainingBackend by registered name."""
        backend_name = name.lower()
        if backend_name not in cls._registry:
            available = ", ".join(cls.list_available())
            raise ValueError(
                f"Training backend '{name}' is not registered. Available backends: [{available}]"
            )
        return cls._registry[backend_name]()

    @classmethod
    def list_available(cls) -> list[str]:
        """List names of all registered backends."""
        return sorted(list(cls._registry.keys()))
