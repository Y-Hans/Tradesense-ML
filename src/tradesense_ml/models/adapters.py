"""PEFT and Adapter loading abstractions."""

from enum import Enum

from pydantic import BaseModel, Field


class AdapterType(str, Enum):
    """Adapter technology type."""

    LORA = "LORA"
    QLORA = "QLORA"
    PROMPT_TUNING = "PROMPT_TUNING"
    FULL_FINE_TUNE = "FULL_FINE_TUNE"


class AdapterConfig(BaseModel):
    """Adapter configuration definition."""

    adapter_type: AdapterType = Field(default=AdapterType.LORA)
    rank: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    bias: str = Field(default="none")
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
