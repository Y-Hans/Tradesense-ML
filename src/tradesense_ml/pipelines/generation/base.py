"""Synthetic dataset generation pipeline abstractions."""

from abc import ABC, abstractmethod
from typing import Any

from tradesense_ml.domain.schemas.market_context import MarketContext
from tradesense_ml.domain.schemas.synthetic import (
    BiasInjectionConfig,
    MarketScenarioConfig,
    SyntheticGenerationBatch,
)
from tradesense_ml.domain.schemas.trade import Trade
from tradesense_ml.pipelines.base import BasePipeline


class BaseMarketGenerator(ABC):
    """Generator interface for synthetic market context & price series."""

    @abstractmethod
    def generate_market(self, config: MarketScenarioConfig) -> MarketContext:
        """Generate synthetic market context."""
        pass


class BaseTradeSimulator(ABC):
    """Simulator interface for trade executions."""

    @abstractmethod
    def simulate_trade(self, context: MarketContext, user_id: str) -> Trade:
        """Simulate realistic trade entry/exit against market context."""
        pass


class BaseBiasInjector(ABC):
    """Behavioral bias injector into trade series."""

    @abstractmethod
    def inject_bias(self, trade: Trade, bias_config: BiasInjectionConfig) -> Trade:
        """Modify trade parameters to reflect specific behavioral bias."""
        pass


class SyntheticGenerationPipeline(
    BasePipeline[SyntheticGenerationBatch, list[tuple[Trade, MarketContext]]], ABC
):
    """Synthetic dataset generation orchestrator pipeline."""

    def __init__(self) -> None:
        super().__init__(pipeline_name="synthetic_generation_pipeline")

    @abstractmethod
    def run(
        self, input_data: SyntheticGenerationBatch, **kwargs: Any
    ) -> list[tuple[Trade, MarketContext]]:
        """Run batch generation of market scenarios and trades."""
        pass
