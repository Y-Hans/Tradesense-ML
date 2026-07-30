"""Fine-tuning trainer orchestrator interface."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from tradesense_ml.models.adapters import AdapterConfig
from tradesense_ml.pipelines.base import BasePipeline


class TrainingConfig(BaseModel):
    """Training run configuration parameters."""

    run_name: str = Field(..., description="Unique training run identifier")
    base_model_name: str = Field(..., description="HF base model identifier")
    dataset_version: str = Field(..., description="Version of training dataset to train on")
    output_dir: str = Field(default="./outputs/checkpoints", description="Checkpoint directory")
    adapter_config: AdapterConfig = Field(default_factory=AdapterConfig)
    learning_rate: float = Field(default=2e-4, gt=0)
    batch_size: int = Field(default=4, gt=0)
    num_epochs: int = Field(default=3, gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0.0)


class BaseTrainingPipeline(BasePipeline[TrainingConfig, str], ABC):
    """Abstract interface for fine-tuning student models."""

    def __init__(self) -> None:
        super().__init__(pipeline_name="training_pipeline")

    @abstractmethod
    def run(self, input_data: TrainingConfig, **kwargs: Any) -> str:
        """Run fine-tuning and return checkpoint directory path."""
        pass
