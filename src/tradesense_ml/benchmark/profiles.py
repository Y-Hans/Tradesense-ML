"""BenchmarkProfile registry and declarative objective configuration definitions."""

from typing import ClassVar

from tradesense_ml.domain.schemas.benchmark import BenchmarkProfile


class BaseProfileBuilder:
    """Abstract builder for declarative benchmark profiles."""

    profile_id: str
    name: str
    description: str

    @classmethod
    def build(cls) -> BenchmarkProfile:
        raise NotImplementedError


class TeacherEvaluationProfile(BaseProfileBuilder):
    """Profile for evaluating teacher model coaching, risk, discipline, reasoning, actionability, and safety."""

    profile_id = "teacher_evaluation"
    name = "Teacher Model Evaluation Profile"
    description = "Comprehensive evaluation profile for teacher models across coaching, risk, discipline, actionability, reasoning, and safety."

    @classmethod
    def build(cls) -> BenchmarkProfile:
        return BenchmarkProfile(
            profile_id=cls.profile_id,
            name=cls.name,
            description=cls.description,
            suite_names=["teacher_benchmark_suite"],
            enabled_case_ids=[
                "coaching_quality",
                "risk_analysis",
                "discipline_analysis",
                "educational_quality",
                "actionability",
                "reasoning_quality",
                "safety",
            ],
            enabled_metric_ids=[
                "quality_score",
                "pass_rate",
                "consistency_score",
                "confidence",
                "latency",
                "token_usage",
            ],
            category_weights={
                "coaching": 0.3,
                "risk_discipline": 0.3,
                "reasoning_actionability": 0.25,
                "safety_compliance": 0.15,
            },
            case_weights={
                "coaching_quality": 1.5,
                "risk_analysis": 1.5,
                "discipline_analysis": 1.5,
                "educational_quality": 1.0,
                "actionability": 1.2,
                "reasoning_quality": 1.3,
                "safety": 2.0,
            },
            execution_policy={"retries": 1, "seed": 42, "fail_fast": False},
        )


class DatasetQualityProfile(BaseProfileBuilder):
    """Profile for evaluating dataset health, consistency, completeness, and prompt adherence."""

    profile_id = "dataset_quality"
    name = "Dataset Quality Profile"
    description = "Evaluates dataset cleanliness, internal/factual consistency, completeness, prompt adherence, and safety."

    @classmethod
    def build(cls) -> BenchmarkProfile:
        return BenchmarkProfile(
            profile_id=cls.profile_id,
            name=cls.name,
            description=cls.description,
            suite_names=["dataset_benchmark_suite"],
            enabled_case_ids=[
                "internal_consistency",
                "factual_consistency",
                "completeness",
                "prompt_adherence",
                "safety",
            ],
            enabled_metric_ids=[
                "accuracy",
                "pass_rate",
                "quality_score",
                "consistency_score",
                "response_length",
                "prompt_length",
            ],
            category_weights={
                "consistency": 0.4,
                "completeness_adherence": 0.4,
                "safety_compliance": 0.2,
            },
            case_weights={
                "internal_consistency": 1.2,
                "factual_consistency": 1.5,
                "completeness": 1.2,
                "prompt_adherence": 1.1,
                "safety": 1.5,
            },
            execution_policy={"retries": 0, "seed": 42, "fail_fast": False},
        )


class PromptEvaluationProfile(BaseProfileBuilder):
    """Profile for evaluating prompt template adherence, token usage, and formatting compliance."""

    profile_id = "prompt_evaluation"
    name = "Prompt Template Evaluation Profile"
    description = "Measures prompt template efficiency, token usage, instruction adherence, and response formatting."

    @classmethod
    def build(cls) -> BenchmarkProfile:
        return BenchmarkProfile(
            profile_id=cls.profile_id,
            name=cls.name,
            description=cls.description,
            suite_names=["prompt_benchmark_suite"],
            enabled_case_ids=[
                "prompt_adherence",
                "completeness",
                "actionability",
            ],
            enabled_metric_ids=[
                "pass_rate",
                "token_usage",
                "cost",
                "prompt_length",
                "response_length",
            ],
            category_weights={
                "formatting_compliance": 0.5,
                "efficiency": 0.5,
            },
            case_weights={
                "prompt_adherence": 2.0,
                "completeness": 1.0,
                "actionability": 1.0,
            },
            execution_policy={"retries": 1, "seed": 42, "fail_fast": False},
        )


class ProfileRegistry:
    """Central registry for benchmark profiles."""

    _registry: ClassVar[dict[str, type[BaseProfileBuilder]]] = {}

    @classmethod
    def register(cls, profile_builder: type[BaseProfileBuilder]) -> type[BaseProfileBuilder]:
        """Register a profile builder."""
        cls._registry[profile_builder.profile_id] = profile_builder
        return profile_builder

    @classmethod
    def get(cls, profile_id: str) -> BenchmarkProfile:
        """Get benchmark profile by ID."""
        if profile_id not in cls._registry:
            raise KeyError(
                f"Benchmark profile '{profile_id}' not found in ProfileRegistry. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[profile_id].build()

    @classmethod
    def list_profiles(cls) -> list[str]:
        """List registered profile IDs."""
        return list(cls._registry.keys())


# Register default built-in profiles
ProfileRegistry.register(TeacherEvaluationProfile)
ProfileRegistry.register(DatasetQualityProfile)
ProfileRegistry.register(PromptEvaluationProfile)
