"""Distillation Strategy abstraction and strategy registry encapsulating recipe execution."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from tradesense_ml.distillation.curriculum import CurriculumBuilder
from tradesense_ml.distillation.filtering import FilteringEngine
from tradesense_ml.distillation.preference import PreferenceBuilder
from tradesense_ml.distillation.sampling import SamplingEngine
from tradesense_ml.distillation.selection import SelectionEngine
from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.dataset import DatasetArtifact
from tradesense_ml.domain.schemas.distillation import DistillationProcessingResult
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class BaseDistillationStrategy(ABC):
    """Abstract interface for distillation pipeline execution strategies."""

    strategy_name: str = "base"

    def __init__(self) -> None:
        self.selection_engine = SelectionEngine()
        self.filtering_engine = FilteringEngine()
        self.sampling_engine = SamplingEngine()
        self.curriculum_builder = CurriculumBuilder()
        self.preference_builder = PreferenceBuilder()

    @abstractmethod
    def execute(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        config_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DistillationProcessingResult:
        """Execute distillation recipe and produce DistillationProcessingResult."""
        pass


class SFTStrategy(BaseDistillationStrategy):
    """Supervised Fine-Tuning focus strategy: Selection -> Filtering -> Sampling -> Curriculum."""

    strategy_name = "SFTStrategy"

    def execute(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        config_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DistillationProcessingResult:
        opts = config_dict or {}

        sel_strat = str(opts.get("selection_strategy", "ThresholdSelection"))
        threshold = float(opts.get("selection_threshold", kwargs.get("threshold", 7.0)))
        top_k = opts.get("top_k", kwargs.get("top_k"))
        seed = int(opts.get("random_seed", kwargs.get("seed", 42)))

        samp_strat = str(opts.get("sampling_strategy", "UniformSampling"))
        sample_size = opts.get("sample_size")
        sampling_rate = float(opts.get("sampling_rate", 1.0))

        curr_strat = str(opts.get("curriculum_strategy", "StandardCurriculumStrategy"))

        # 1. Selection
        sel_result = self.selection_engine.select(
            dataset_artifact=dataset_artifact,
            benchmark_artifact=benchmark_artifact,
            strategy_name=sel_strat,
            threshold=threshold,
            top_k=top_k,
            seed=seed,
        )

        # 2. Filtering
        passed_ex, rejected_ex, filter_stats = self.filtering_engine.filter_examples(
            examples=sel_result.selected_examples,
            benchmark_artifact=benchmark_artifact,
            min_quality_score=threshold,
        )

        # Combine selection rejected with filtering rejected
        all_rejected = sel_result.selected_examples[len(passed_ex) :] + [
            _for_ex for _for_ex in sel_result.selected_examples if _for_ex not in passed_ex
        ]

        # 3. Sampling
        samp_result = self.sampling_engine.sample(
            examples=passed_ex,
            strategy_name=samp_strat,
            sample_size=sample_size,
            sampling_rate=sampling_rate,
            seed=seed,
        )

        # 4. Curriculum
        stages = self.curriculum_builder.build_curriculum(
            examples=samp_result.sampled_examples,
            benchmark_artifact=benchmark_artifact,
            strategy_name=curr_strat,
        )

        return DistillationProcessingResult(
            selected_examples=passed_ex,
            rejected_examples=all_rejected,
            sampled_examples=samp_result.sampled_examples,
            curriculum_stages=stages,
            preference_pairs=[],
            selection_result=sel_result,
            sampling_result=samp_result,
            filtering_stats=filter_stats,
            processing_metadata={"strategy_name": self.strategy_name},
        )


class DPOStrategy(BaseDistillationStrategy):
    """Direct Preference Optimization strategy: Selection -> Filtering -> Preference Pair Builder."""

    strategy_name = "DPOStrategy"

    def execute(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        config_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DistillationProcessingResult:
        opts = config_dict or {}
        threshold = float(opts.get("selection_threshold", kwargs.get("threshold", 6.5)))
        seed = int(opts.get("random_seed", kwargs.get("seed", 42)))

        sel_result = self.selection_engine.select(
            dataset_artifact=dataset_artifact,
            benchmark_artifact=benchmark_artifact,
            strategy_name="ThresholdSelection",
            threshold=threshold,
            seed=seed,
        )

        passed_ex, rejected_ex, filter_stats = self.filtering_engine.filter_examples(
            examples=sel_result.selected_examples,
            benchmark_artifact=benchmark_artifact,
            min_quality_score=threshold,
        )

        pref_pairs = self.preference_builder.build_preference_pairs(
            chosen_examples=passed_ex,
            rejected_examples=rejected_ex,
            benchmark_artifact=benchmark_artifact,
        )

        samp_result = self.sampling_engine.sample(
            examples=passed_ex,
            strategy_name="UniformSampling",
            seed=seed,
        )

        stages = self.curriculum_builder.build_curriculum(
            examples=passed_ex, benchmark_artifact=benchmark_artifact
        )

        return DistillationProcessingResult(
            selected_examples=passed_ex,
            rejected_examples=rejected_ex,
            sampled_examples=samp_result.sampled_examples,
            curriculum_stages=stages,
            preference_pairs=pref_pairs,
            selection_result=sel_result,
            sampling_result=samp_result,
            filtering_stats=filter_stats,
            processing_metadata={
                "strategy_name": self.strategy_name,
                "pairs_count": len(pref_pairs),
            },
        )


class ORPOStrategy(BaseDistillationStrategy):
    """Odds-Ratio Preference Optimization strategy."""

    strategy_name = "ORPOStrategy"

    def execute(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        config_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DistillationProcessingResult:
        # Same flow as DPOStrategy with ORPO metadata
        res = DPOStrategy().execute(dataset_artifact, benchmark_artifact, config_dict, **kwargs)
        return DistillationProcessingResult(
            selected_examples=res.selected_examples,
            rejected_examples=res.rejected_examples,
            sampled_examples=res.sampled_examples,
            curriculum_stages=res.curriculum_stages,
            preference_pairs=res.preference_pairs,
            selection_result=res.selection_result,
            sampling_result=res.sampling_result,
            filtering_stats=res.filtering_stats,
            processing_metadata={"strategy_name": self.strategy_name, "orpo_enabled": True},
        )


class CurriculumStrategy(BaseDistillationStrategy):
    """Multi-stage curriculum building strategy."""

    strategy_name = "CurriculumStrategy"

    def execute(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        config_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DistillationProcessingResult:
        return SFTStrategy().execute(dataset_artifact, benchmark_artifact, config_dict, **kwargs)


class HybridStrategy(BaseDistillationStrategy):
    """Comprehensive strategy combining SFT examples, Preference pairs, and Curriculum stages."""

    strategy_name = "HybridStrategy"

    def execute(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        config_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DistillationProcessingResult:
        opts = config_dict or {}
        threshold = float(opts.get("selection_threshold", kwargs.get("threshold", 7.0)))
        seed = int(opts.get("random_seed", kwargs.get("seed", 42)))

        sel_result = self.selection_engine.select(
            dataset_artifact=dataset_artifact,
            benchmark_artifact=benchmark_artifact,
            strategy_name="TopScoreSelection",
            threshold=threshold,
            seed=seed,
        )

        passed_ex, rejected_ex, filter_stats = self.filtering_engine.filter_examples(
            examples=sel_result.selected_examples,
            benchmark_artifact=benchmark_artifact,
            min_quality_score=threshold,
        )

        samp_result = self.sampling_engine.sample(
            examples=passed_ex,
            strategy_name="UniformSampling",
            seed=seed,
        )

        stages = self.curriculum_builder.build_curriculum(
            examples=samp_result.sampled_examples,
            benchmark_artifact=benchmark_artifact,
        )

        pref_pairs = self.preference_builder.build_preference_pairs(
            chosen_examples=samp_result.sampled_examples,
            rejected_examples=rejected_ex,
            benchmark_artifact=benchmark_artifact,
        )

        return DistillationProcessingResult(
            selected_examples=passed_ex,
            rejected_examples=rejected_ex,
            sampled_examples=samp_result.sampled_examples,
            curriculum_stages=stages,
            preference_pairs=pref_pairs,
            selection_result=sel_result,
            sampling_result=samp_result,
            filtering_stats=filter_stats,
            processing_metadata={"strategy_name": self.strategy_name, "hybrid_mode": True},
        )


class DistillationStrategyRegistry:
    """Registry for distillation execution strategies."""

    _registry: ClassVar[dict[str, type[BaseDistillationStrategy]]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: type[BaseDistillationStrategy]) -> None:
        cls._registry[name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> BaseDistillationStrategy:
        if name not in cls._registry:
            raise KeyError(
                f"Distillation strategy '{name}' not found. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]()

    @classmethod
    def list_strategies(cls) -> list[str]:
        return list(cls._registry.keys())


# Register built-in distillation strategies
DistillationStrategyRegistry.register("SFTStrategy", SFTStrategy)
DistillationStrategyRegistry.register("DPOStrategy", DPOStrategy)
DistillationStrategyRegistry.register("ORPOStrategy", ORPOStrategy)
DistillationStrategyRegistry.register("CurriculumStrategy", CurriculumStrategy)
DistillationStrategyRegistry.register("HybridStrategy", HybridStrategy)
