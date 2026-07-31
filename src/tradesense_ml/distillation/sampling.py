"""Sampling Engine and pluggable sampling strategies for distillation dataset creation."""

import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from tradesense_ml.domain.schemas.distillation import DistillationExample, SamplingResult
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class BaseSamplingStrategy(ABC):
    """Abstract base class for sampling strategies."""

    strategy_name: str = "base"

    @abstractmethod
    def sample(
        self,
        examples: list[DistillationExample],
        sample_size: int | None = None,
        sampling_rate: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> SamplingResult:
        """Execute sampling over a list of examples."""
        pass


class UniformSampling(BaseSamplingStrategy):
    """Uniform random sampling strategy."""

    strategy_name = "UniformSampling"

    def sample(
        self,
        examples: list[DistillationExample],
        sample_size: int | None = None,
        sampling_rate: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> SamplingResult:
        rng = random.Random(seed)

        target_size = len(examples)
        if sample_size is not None:
            target_size = min(sample_size, len(examples))
        elif sampling_rate < 1.0:
            target_size = max(1, int(len(examples) * sampling_rate))

        if target_size >= len(examples):
            sampled = list(examples)
        else:
            sampled = rng.sample(examples, target_size)

        sampled_ids = [e.example_id for e in sampled]

        return SamplingResult(
            sampled_example_ids=sampled_ids,
            sampled_examples=sampled,
            strategy_name=self.strategy_name,
            sample_size=len(sampled),
            sampling_rate=len(sampled) / max(1, len(examples)),
            distribution_stats={"total_input": len(examples), "sampled": len(sampled)},
            metadata={"seed": seed, "target_size": target_size},
        )


class WeightedSampling(BaseSamplingStrategy):
    """Quality-score weighted random sampling strategy."""

    strategy_name = "WeightedSampling"

    def sample(
        self,
        examples: list[DistillationExample],
        sample_size: int | None = None,
        sampling_rate: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> SamplingResult:
        rng = random.Random(seed)
        if not examples:
            return SamplingResult(
                sampled_example_ids=[],
                sampled_examples=[],
                strategy_name=self.strategy_name,
                sample_size=0,
                sampling_rate=0.0,
            )

        target_size = len(examples)
        if sample_size is not None:
            target_size = min(sample_size, len(examples))
        elif sampling_rate < 1.0:
            target_size = max(1, int(len(examples) * sampling_rate))

        weights = [max(0.1, e.quality_score) for e in examples]
        choices = rng.choices(examples, weights=weights, k=target_size)

        seen = set()
        sampled: list[DistillationExample] = []
        for e in choices:
            if e.example_id not in seen:
                seen.add(e.example_id)
                sampled.append(e)

        sampled_ids = [e.example_id for e in sampled]

        return SamplingResult(
            sampled_example_ids=sampled_ids,
            sampled_examples=sampled,
            strategy_name=self.strategy_name,
            sample_size=len(sampled),
            sampling_rate=len(sampled) / max(1, len(examples)),
            distribution_stats={"total_input": len(examples), "sampled": len(sampled)},
            metadata={"seed": seed, "target_size": target_size},
        )


class BalancedSampling(BaseSamplingStrategy):
    """Stratified sampling across difficulty tiers or teacher models."""

    strategy_name = "BalancedSampling"

    def sample(
        self,
        examples: list[DistillationExample],
        sample_size: int | None = None,
        sampling_rate: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> SamplingResult:
        rng = random.Random(seed)
        if not examples:
            return SamplingResult(
                sampled_example_ids=[],
                sampled_examples=[],
                strategy_name=self.strategy_name,
                sample_size=0,
                sampling_rate=0.0,
            )

        # Stratify by quality tier
        by_tier: dict[str, list[DistillationExample]] = {}
        for ex in examples:
            by_tier.setdefault(ex.quality_tier, []).append(ex)

        target_total = (
            min(sample_size, len(examples))
            if sample_size is not None
            else max(1, int(len(examples) * sampling_rate))
        )
        per_tier_target = max(1, target_total // max(1, len(by_tier)))

        sampled: list[DistillationExample] = []
        for tier, items in by_tier.items():
            k = min(per_tier_target, len(items))
            sampled.extend(rng.sample(items, k))

        sampled_ids = [e.example_id for e in sampled]

        return SamplingResult(
            sampled_example_ids=sampled_ids,
            sampled_examples=sampled,
            strategy_name=self.strategy_name,
            sample_size=len(sampled),
            sampling_rate=len(sampled) / max(1, len(examples)),
            distribution_stats={tier: len(items) for tier, items in by_tier.items()},
            metadata={"tiers_count": len(by_tier), "seed": seed},
        )


class CurriculumSampling(BaseSamplingStrategy):
    """Difficulty-oriented sampling prioritizing higher difficulty examples."""

    strategy_name = "CurriculumSampling"

    def sample(
        self,
        examples: list[DistillationExample],
        sample_size: int | None = None,
        sampling_rate: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> SamplingResult:
        if not examples:
            return SamplingResult(
                sampled_example_ids=[],
                sampled_examples=[],
                strategy_name=self.strategy_name,
                sample_size=0,
                sampling_rate=0.0,
            )

        # Sort by difficulty ascending or descending
        sorted_examples = sorted(examples, key=lambda x: x.difficulty)

        target_size = len(examples)
        if sample_size is not None:
            target_size = min(sample_size, len(examples))
        elif sampling_rate < 1.0:
            target_size = max(1, int(len(examples) * sampling_rate))

        sampled = sorted_examples[:target_size]
        sampled_ids = [e.example_id for e in sampled]

        return SamplingResult(
            sampled_example_ids=sampled_ids,
            sampled_examples=sampled,
            strategy_name=self.strategy_name,
            sample_size=len(sampled),
            sampling_rate=len(sampled) / max(1, len(examples)),
            distribution_stats={"total_input": len(examples), "sampled": len(sampled)},
            metadata={"seed": seed, "target_size": target_size},
        )


class RandomDeterministicSampling(BaseSamplingStrategy):
    """Pure seed-deterministic random sampling."""

    strategy_name = "RandomDeterministicSampling"

    def sample(
        self,
        examples: list[DistillationExample],
        sample_size: int | None = None,
        sampling_rate: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> SamplingResult:
        return UniformSampling().sample(
            examples=examples,
            sample_size=sample_size,
            sampling_rate=sampling_rate,
            seed=seed,
            **kwargs,
        )


class SamplingStrategyRegistry:
    """Registry for pluggable sampling strategies."""

    _registry: ClassVar[dict[str, type[BaseSamplingStrategy]]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: type[BaseSamplingStrategy]) -> None:
        cls._registry[name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> BaseSamplingStrategy:
        if name not in cls._registry:
            raise KeyError(
                f"Sampling strategy '{name}' not found. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]()

    @classmethod
    def list_strategies(cls) -> list[str]:
        return list(cls._registry.keys())


# Register built-in sampling strategies
SamplingStrategyRegistry.register("UniformSampling", UniformSampling)
SamplingStrategyRegistry.register("WeightedSampling", WeightedSampling)
SamplingStrategyRegistry.register("BalancedSampling", BalancedSampling)
SamplingStrategyRegistry.register("CurriculumSampling", CurriculumSampling)
SamplingStrategyRegistry.register("RandomDeterministicSampling", RandomDeterministicSampling)


class SamplingEngine:
    """Sampling Engine executing configured sampling strategy."""

    def __init__(self, default_strategy: str = "UniformSampling") -> None:
        self.default_strategy = default_strategy

    def sample(
        self,
        examples: list[DistillationExample],
        strategy_name: str | None = None,
        sample_size: int | None = None,
        sampling_rate: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> SamplingResult:
        st_name = strategy_name or self.default_strategy
        strategy = SamplingStrategyRegistry.get(st_name)
        logger.info(f"Executing SamplingEngine with strategy '{st_name}' (seed={seed})")
        return strategy.sample(
            examples=examples,
            sample_size=sample_size,
            sampling_rate=sampling_rate,
            seed=seed,
            **kwargs,
        )
