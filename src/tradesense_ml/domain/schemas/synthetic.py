"""Synthetic market and trade generation schemas."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.market_context import MarketRegime, VolatilityLevel


class BiasType(str, Enum):
    """Trading behavioral biases for synthetic injection."""

    FOMO = "FOMO"
    REVENGE_TRADING = "REVENGE_TRADING"
    OVERCONFIDENCE = "OVERCONFIDENCE"
    LOSS_AVERSION = "LOSS_AVERSION"
    DISPOSITION_EFFECT = "DISPOSITION_EFFECT"
    NO_BIAS_CLEAN = "NO_BIAS_CLEAN"


class MarketScenarioConfig(BaseModel):
    """Configuration parameter for generating a synthetic market regime."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str = Field(..., description="Unique scenario ID")
    symbol: str = Field(..., description="Target symbol")
    regime: MarketRegime = Field(..., description="Target regime")
    volatility: VolatilityLevel = Field(..., description="Target volatility level")
    num_candles: int = Field(default=100, gt=0, description="Length of price series")
    seed: int | None = Field(default=42, description="Random seed for reproducibility")


class BiasInjectionConfig(BaseModel):
    """Parameters for injecting cognitive behavioral biases into trade histories."""

    model_config = ConfigDict(frozen=True)

    bias_type: BiasType = Field(..., description="Type of cognitive bias to simulate")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="Severity of bias (0-1)")
    description: str = Field(..., description="Description of behavioral deviation")


class SyntheticGenerationBatch(BaseModel):
    """Batch metadata container for synthetic dataset generation jobs."""

    model_config = ConfigDict(frozen=True)

    batch_id: str = Field(..., description="Unique batch generation ID")
    generator_version: str = Field(..., description="Generator version identifier")
    scenario_configs: list[MarketScenarioConfig] = Field(
        ..., description="Market scenarios generated"
    )
    bias_configs: list[BiasInjectionConfig] = Field(..., description="Bias configurations applied")
    total_samples: int = Field(..., gt=0, description="Number of synthetic samples requested")
    generation_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Job configuration parameters"
    )
