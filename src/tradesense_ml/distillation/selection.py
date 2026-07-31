"""Selection Engine and pluggable selection strategies for teacher output selection."""

import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.dataset import DatasetArtifact, DatasetExample
from tradesense_ml.domain.schemas.distillation import DistillationExample, SelectionResult
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


def _convert_dataset_example(
    example: DatasetExample, default_teacher: str = "teacher_llm_v1"
) -> DistillationExample:
    """Helper to convert DatasetExample to DistillationExample."""
    review_info = example.review_info or {}
    quality_score = float(review_info.get("quality_score", 8.0))

    # Derive difficulty from review_info or metadata or fallback
    diff = float(example.metadata.get("difficulty", 0.5))
    tier = "medium"
    if diff <= 0.25:
        tier = "easy"
    elif diff <= 0.55:
        tier = "medium"
    elif diff <= 0.8:
        tier = "hard"
    else:
        tier = "expert"

    teacher_id = str(
        example.metadata.get("teacher_id", review_info.get("teacher_id", default_teacher))
    )

    return DistillationExample(
        example_id=example.example_id,
        instruction=example.instruction,
        input=example.input,
        output=example.output,
        prompt=example.prompt,
        messages=example.messages,
        reasoning=example.reasoning,
        quality_score=quality_score,
        difficulty=diff,
        quality_tier=tier,
        teacher_id=teacher_id,
        format_type=example.format_type,
        metadata=example.metadata,
    )


class BaseSelectionStrategy(ABC):
    """Abstract base class for selection strategies."""

    strategy_name: str = "base"

    @abstractmethod
    def select(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        threshold: float = 7.0,
        top_k: int | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> SelectionResult:
        """Execute example selection."""
        pass


class ThresholdSelection(BaseSelectionStrategy):
    """Selects examples meeting or exceeding a minimum quality threshold."""

    strategy_name = "ThresholdSelection"

    def select(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        threshold: float = 7.0,
        top_k: int | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> SelectionResult:
        selected_examples: list[DistillationExample] = []
        selected_ids: list[str] = []
        rejected_ids: list[str] = []
        scores_map: dict[str, float] = {}

        # Extract all examples from splits
        all_examples: list[DatasetExample] = []
        for split_examples in dataset_artifact.splits.values():
            all_examples.extend(split_examples)

        teacher_id = kwargs.get("teacher_id", "teacher_llm_v1")

        for ex in all_examples:
            dist_ex = _convert_dataset_example(ex, default_teacher=teacher_id)
            scores_map[ex.example_id] = dist_ex.quality_score

            if dist_ex.quality_score >= threshold:
                selected_examples.append(dist_ex)
                selected_ids.append(ex.example_id)
            else:
                rejected_ids.append(ex.example_id)

        if top_k is not None and len(selected_examples) > top_k:
            selected_examples.sort(key=lambda x: x.quality_score, reverse=True)
            trimmed_selected = selected_examples[:top_k]
            rejected_ids.extend([e.example_id for e in selected_examples[top_k:]])
            selected_examples = trimmed_selected
            selected_ids = [e.example_id for e in selected_examples]

        counts = {
            "total_input": len(all_examples),
            "selected_count": len(selected_ids),
            "rejected_count": len(rejected_ids),
        }

        return SelectionResult(
            selected_example_ids=selected_ids,
            selected_examples=selected_examples,
            rejected_example_ids=rejected_ids,
            strategy_name=self.strategy_name,
            selection_counts=counts,
            threshold_applied=threshold,
            scores_map=scores_map,
            metadata={"threshold": threshold, "top_k": top_k},
        )


class TopScoreSelection(BaseSelectionStrategy):
    """Selects top N highest scoring examples."""

    strategy_name = "TopScoreSelection"

    def select(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        threshold: float = 0.0,
        top_k: int | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> SelectionResult:
        all_examples: list[DatasetExample] = []
        for split_examples in dataset_artifact.splits.values():
            all_examples.extend(split_examples)

        converted = [_convert_dataset_example(ex) for ex in all_examples]
        scores_map = {e.example_id: e.quality_score for e in converted}

        # Filter by threshold if non-zero
        valid_examples = [e for e in converted if e.quality_score >= threshold]
        valid_examples.sort(key=lambda x: x.quality_score, reverse=True)

        k = top_k if top_k is not None else len(valid_examples)
        selected = valid_examples[:k]
        rejected = [e for e in converted if e.example_id not in {s.example_id for s in selected}]

        selected_ids = [e.example_id for e in selected]
        rejected_ids = [e.example_id for e in rejected]

        return SelectionResult(
            selected_example_ids=selected_ids,
            selected_examples=selected,
            rejected_example_ids=rejected_ids,
            strategy_name=self.strategy_name,
            selection_counts={
                "total_input": len(all_examples),
                "selected_count": len(selected_ids),
                "rejected_count": len(rejected_ids),
            },
            threshold_applied=threshold,
            scores_map=scores_map,
            metadata={"top_k": k},
        )


class BalancedSelection(BaseSelectionStrategy):
    """Selects balanced number of examples across teacher models or quality tiers."""

    strategy_name = "BalancedSelection"

    def select(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        threshold: float = 7.0,
        top_k: int | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> SelectionResult:
        all_examples: list[DatasetExample] = []
        for split_examples in dataset_artifact.splits.values():
            all_examples.extend(split_examples)

        converted = [_convert_dataset_example(ex) for ex in all_examples]
        scores_map = {e.example_id: e.quality_score for e in converted}

        # Group by teacher_id
        grouped: dict[str, list[DistillationExample]] = {}
        for ex in converted:
            if ex.quality_score >= threshold:
                grouped.setdefault(ex.teacher_id, []).append(ex)

        selected: list[DistillationExample] = []
        for teacher_id, items in grouped.items():
            items.sort(key=lambda x: x.quality_score, reverse=True)
            k_teacher = top_k // max(1, len(grouped)) if top_k else len(items)
            selected.extend(items[:k_teacher])

        selected_set = {e.example_id for e in selected}
        rejected = [e for e in converted if e.example_id not in selected_set]

        return SelectionResult(
            selected_example_ids=[e.example_id for e in selected],
            selected_examples=selected,
            rejected_example_ids=[e.example_id for e in rejected],
            strategy_name=self.strategy_name,
            selection_counts={
                "total_input": len(all_examples),
                "selected_count": len(selected),
                "rejected_count": len(rejected),
                "teacher_groups": len(grouped),
            },
            threshold_applied=threshold,
            scores_map=scores_map,
            metadata={"teachers": list(grouped.keys())},
        )


class WeightedSelection(BaseSelectionStrategy):
    """Probabilistic selection weighted by quality scores."""

    strategy_name = "WeightedSelection"

    def select(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        threshold: float = 6.0,
        top_k: int | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> SelectionResult:
        rng = random.Random(seed)
        all_examples: list[DatasetExample] = []
        for split_examples in dataset_artifact.splits.values():
            all_examples.extend(split_examples)

        converted = [_convert_dataset_example(ex) for ex in all_examples]
        scores_map = {e.example_id: e.quality_score for e in converted}

        candidates = [e for e in converted if e.quality_score >= threshold]
        if not candidates:
            candidates = converted

        k = top_k if top_k is not None else len(candidates)
        k = min(k, len(candidates))

        weights = [max(0.1, e.quality_score) for e in candidates]
        selected = rng.choices(candidates, weights=weights, k=k) if candidates else []

        # Deduplicate while preserving selected order
        seen = set()
        deduped_selected: list[DistillationExample] = []
        for e in selected:
            if e.example_id not in seen:
                seen.add(e.example_id)
                deduped_selected.append(e)

        selected_set = {e.example_id for e in deduped_selected}
        rejected = [e for e in converted if e.example_id not in selected_set]

        return SelectionResult(
            selected_example_ids=[e.example_id for e in deduped_selected],
            selected_examples=deduped_selected,
            rejected_example_ids=[e.example_id for e in rejected],
            strategy_name=self.strategy_name,
            selection_counts={
                "total_input": len(all_examples),
                "selected_count": len(deduped_selected),
                "rejected_count": len(rejected),
            },
            threshold_applied=threshold,
            scores_map=scores_map,
            metadata={"seed": seed, "top_k": k},
        )


class SelectionStrategyRegistry:
    """Registry for selection strategies."""

    _registry: ClassVar[dict[str, type[BaseSelectionStrategy]]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: type[BaseSelectionStrategy]) -> None:
        """Register a strategy class."""
        cls._registry[name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> BaseSelectionStrategy:
        """Instantiate and return registered strategy."""
        if name not in cls._registry:
            raise KeyError(
                f"Selection strategy '{name}' not found. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]()

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List registered strategy names."""
        return list(cls._registry.keys())


# Register built-in selection strategies
SelectionStrategyRegistry.register("ThresholdSelection", ThresholdSelection)
SelectionStrategyRegistry.register("TopScoreSelection", TopScoreSelection)
SelectionStrategyRegistry.register("BalancedSelection", BalancedSelection)
SelectionStrategyRegistry.register("WeightedSelection", WeightedSelection)


class SelectionEngine:
    """Selection Engine orchestrating teacher output selection using pluggable strategies."""

    def __init__(self, default_strategy: str = "ThresholdSelection") -> None:
        self.default_strategy = default_strategy

    def select(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        strategy_name: str | None = None,
        threshold: float = 7.0,
        top_k: int | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> SelectionResult:
        st_name = strategy_name or self.default_strategy
        strategy = SelectionStrategyRegistry.get(st_name)
        logger.info(
            f"Executing SelectionEngine with strategy '{st_name}' (threshold={threshold}, seed={seed})"
        )
        return strategy.select(
            dataset_artifact=dataset_artifact,
            benchmark_artifact=benchmark_artifact,
            threshold=threshold,
            top_k=top_k,
            seed=seed,
            **kwargs,
        )
