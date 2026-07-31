"""Curriculum Builder and difficulty-based curriculum strategies."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.distillation import CurriculumStage, DistillationExample
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class BaseCurriculumStrategy(ABC):
    """Abstract base class for curriculum strategies."""

    strategy_name: str = "base"

    @abstractmethod
    def build_curriculum(
        self,
        examples: list[DistillationExample],
        benchmark_artifact: BenchmarkArtifact | None = None,
        **kwargs: Any,
    ) -> list[CurriculumStage]:
        """Build difficulty curriculum stages."""
        pass


class StandardCurriculumStrategy(BaseCurriculumStrategy):
    """Standard 4-tier curriculum strategy: Easy, Medium, Hard, Expert."""

    strategy_name = "StandardCurriculumStrategy"

    def build_curriculum(
        self,
        examples: list[DistillationExample],
        benchmark_artifact: BenchmarkArtifact | None = None,
        **kwargs: Any,
    ) -> list[CurriculumStage]:
        easy_examples: list[DistillationExample] = []
        medium_examples: list[DistillationExample] = []
        hard_examples: list[DistillationExample] = []
        expert_examples: list[DistillationExample] = []

        for ex in examples:
            d = ex.difficulty
            if d <= 0.25:
                easy_examples.append(ex)
            elif d <= 0.55:
                medium_examples.append(ex)
            elif d <= 0.80:
                hard_examples.append(ex)
            else:
                expert_examples.append(ex)

        stages = [
            CurriculumStage(
                stage_id="stage_1_easy",
                name="Easy",
                description="Foundational trade coaching concepts and high-clarity setups",
                stage_order=1,
                min_difficulty=0.0,
                max_difficulty=0.25,
                examples=easy_examples,
                example_ids=[e.example_id for e in easy_examples],
                example_count=len(easy_examples),
            ),
            CurriculumStage(
                stage_id="stage_2_medium",
                name="Medium",
                description="Standard trade executions with clear risk-reward boundaries",
                stage_order=2,
                min_difficulty=0.26,
                max_difficulty=0.55,
                examples=medium_examples,
                example_ids=[e.example_id for e in medium_examples],
                example_count=len(medium_examples),
            ),
            CurriculumStage(
                stage_id="stage_3_hard",
                name="Hard",
                description="Complex market regime shifts and multi-step risk evaluations",
                stage_order=3,
                min_difficulty=0.56,
                max_difficulty=0.80,
                examples=hard_examples,
                example_ids=[e.example_id for e in hard_examples],
                example_count=len(hard_examples),
            ),
            CurriculumStage(
                stage_id="stage_4_expert",
                name="Expert",
                description="High-volatility, subtle behavioral biases, and counter-intuitive market setups",
                stage_order=4,
                min_difficulty=0.81,
                max_difficulty=1.0,
                examples=expert_examples,
                example_ids=[e.example_id for e in expert_examples],
                example_count=len(expert_examples),
            ),
        ]

        logger.info(
            f"StandardCurriculumStrategy built {len(stages)} stages: Easy={len(easy_examples)}, Medium={len(medium_examples)}, Hard={len(hard_examples)}, Expert={len(expert_examples)}"
        )
        return stages


class DifficultyCurriculumStrategy(BaseCurriculumStrategy):
    """Difficulty curriculum derived dynamically from quality scores or benchmark feedback."""

    strategy_name = "DifficultyCurriculumStrategy"

    def build_curriculum(
        self,
        examples: list[DistillationExample],
        benchmark_artifact: BenchmarkArtifact | None = None,
        **kwargs: Any,
    ) -> list[CurriculumStage]:

        # Dynamically partition into N equal quantiles or tiers based on quality score
        sorted_examples = sorted(examples, key=lambda x: x.quality_score, reverse=True)
        total = len(sorted_examples)

        if total == 0:
            return StandardCurriculumStrategy().build_curriculum(examples, benchmark_artifact)

        chunk = max(1, total // 3)

        easy_tier = sorted_examples[:chunk]
        medium_tier = sorted_examples[chunk : chunk * 2]
        hard_tier = sorted_examples[chunk * 2 :]

        return [
            CurriculumStage(
                stage_id="stage_1_foundational",
                name="Easy",
                description="High confidence top quality responses",
                stage_order=1,
                min_difficulty=0.0,
                max_difficulty=0.33,
                examples=easy_tier,
                example_ids=[e.example_id for e in easy_tier],
                example_count=len(easy_tier),
            ),
            CurriculumStage(
                stage_id="stage_2_intermediate",
                name="Medium",
                description="Moderate quality trade coaching scenarios",
                stage_order=2,
                min_difficulty=0.34,
                max_difficulty=0.66,
                examples=medium_tier,
                example_ids=[e.example_id for e in medium_tier],
                example_count=len(medium_tier),
            ),
            CurriculumStage(
                stage_id="stage_3_advanced",
                name="Hard",
                description="Complex or borderline trade coaching scenarios requiring nuance",
                stage_order=3,
                min_difficulty=0.67,
                max_difficulty=1.0,
                examples=hard_tier,
                example_ids=[e.example_id for e in hard_tier],
                example_count=len(hard_tier),
            ),
        ]


class CurriculumStrategyRegistry:
    """Registry for pluggable curriculum strategies."""

    _registry: ClassVar[dict[str, type[BaseCurriculumStrategy]]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: type[BaseCurriculumStrategy]) -> None:
        cls._registry[name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> BaseCurriculumStrategy:
        if name not in cls._registry:
            raise KeyError(
                f"Curriculum strategy '{name}' not found. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]()

    @classmethod
    def list_strategies(cls) -> list[str]:
        return list(cls._registry.keys())


# Register built-in curriculum strategies
CurriculumStrategyRegistry.register("StandardCurriculumStrategy", StandardCurriculumStrategy)
CurriculumStrategyRegistry.register("DifficultyCurriculumStrategy", DifficultyCurriculumStrategy)


class CurriculumBuilder:
    """Curriculum Builder orchestrating curriculum stage generation."""

    def __init__(self, default_strategy: str = "StandardCurriculumStrategy") -> None:
        self.default_strategy = default_strategy

    def build_curriculum(
        self,
        examples: list[DistillationExample],
        benchmark_artifact: BenchmarkArtifact | None = None,
        strategy_name: str | None = None,
        **kwargs: Any,
    ) -> list[CurriculumStage]:
        st_name = strategy_name or self.default_strategy
        strategy = CurriculumStrategyRegistry.get(st_name)
        logger.info(f"Executing CurriculumBuilder with strategy '{st_name}'")
        return strategy.build_curriculum(
            examples=examples, benchmark_artifact=benchmark_artifact, **kwargs
        )
