"""Model Registry abstractions for registering, querying, and managing model assets."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DeploymentStage(str, Enum):
    """Deployment lifecycle stage."""

    EXPERIMENTAL = "EXPERIMENTAL"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"


class ModelMetadata(BaseModel):
    """Model registry record metadata."""

    model_id: str = Field(..., description="Registered model ID, e.g. tradesense-qwen-7b-v1")
    base_model: str = Field(..., description="Base model name, e.g. Qwen/Qwen2.5-7B-Instruct")
    adapter_type: str = Field(default="QLoRA", description="Adapter technique used")
    dataset_version: str = Field(..., description="Dataset version used for training")
    evaluation_scores: dict[str, float] = Field(
        default_factory=dict, description="Benchmark scores"
    )
    deployment_stage: DeploymentStage = Field(
        default=DeploymentStage.EXPERIMENTAL, description="Stage"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    artifact_uri: str = Field(..., description="Path to checkpoint artifact")
    tags: list[str] = Field(default_factory=list)


class BaseModelRegistry(ABC):
    """Abstract interface for model registry backends."""

    @abstractmethod
    def register_model(self, metadata: ModelMetadata) -> None:
        """Register a trained model checkpoint."""
        pass

    @abstractmethod
    def get_model(self, model_id: str) -> ModelMetadata:
        """Retrieve model metadata by ID."""
        pass

    @abstractmethod
    def list_models(self, stage: DeploymentStage | None = None) -> list[ModelMetadata]:
        """List registered models."""
        pass

    @abstractmethod
    def update_stage(self, model_id: str, new_stage: DeploymentStage) -> None:
        """Promote or demote model deployment stage."""
        pass
