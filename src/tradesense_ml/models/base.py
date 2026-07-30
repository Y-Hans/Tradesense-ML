"""Base student model abstractions."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from tradesense_ml.domain.schemas.inference import InferenceRequest, InferenceResponse


class FineTuningStrategy(BaseModel):
    """Fine-tuning strategy configuration."""

    strategy_type: str = Field(..., description="LoRA, QLoRA, or Full")
    r: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Target attention modules",
    )


class BaseStudentModel(ABC):
    """Abstract interface for TradeSense student models."""

    def __init__(self, model_id: str, base_model_name: str) -> None:
        self.model_id = model_id
        self.base_model_name = base_model_name

    @abstractmethod
    def load_weights(self, checkpoint_path: str) -> None:
        """Load weights from a checkpoint path."""
        pass

    @abstractmethod
    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Execute inference for a student model."""
        pass
