"""Pluggable benchmark suite architecture and central suite registry."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from tradesense_ml.benchmark.cases import CaseRegistry
from tradesense_ml.domain.schemas.benchmark import BenchmarkExecutionResult, BenchmarkSuite
from tradesense_ml.domain.schemas.dataset import DatasetArtifact


class BaseBenchmarkSuite(ABC):
    """Abstract pluggable benchmark suite coordinating benchmark cases."""

    suite_id: str
    name: str
    version: str = "v1.0.0"
    description: str = ""
    default_case_ids: list[str] = []

    def get_definition(self) -> BenchmarkSuite:
        """Get canonical BenchmarkSuite model definition."""
        cases = [
            CaseRegistry.get(cid).get_definition()
            for cid in self.default_case_ids
            if cid in CaseRegistry.list_cases()
        ]
        return BenchmarkSuite(
            suite_id=self.suite_id,
            name=self.name,
            version=self.version,
            description=self.description,
            cases=cases,
        )

    @abstractmethod
    def run_cases(
        self,
        dataset_artifact: DatasetArtifact,
        enabled_case_ids: list[str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> list[BenchmarkExecutionResult]:
        """Execute suite's benchmark cases against DatasetArtifact and return un-scored execution results."""
        pass


class SuiteRegistry:
    """Central registry for discovering and registering pluggable benchmark suites."""

    _registry: ClassVar[dict[str, type[BaseBenchmarkSuite]]] = {}

    @classmethod
    def register(cls, suite_cls: type[BaseBenchmarkSuite]) -> type[BaseBenchmarkSuite]:
        """Register a benchmark suite class."""
        cls._registry[suite_cls.suite_id] = suite_cls
        return suite_cls

    @classmethod
    def get(cls, suite_id: str) -> BaseBenchmarkSuite:
        """Instantiate registered benchmark suite by suite_id."""
        if suite_id not in cls._registry:
            raise KeyError(
                f"Benchmark suite '{suite_id}' not found in SuiteRegistry. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[suite_id]()

    @classmethod
    def list_suites(cls) -> list[str]:
        """List registered benchmark suite IDs."""
        return list(cls._registry.keys())


@SuiteRegistry.register
class TeacherBenchmarkSuite(BaseBenchmarkSuite):
    """Suite evaluating teacher model performance across coaching, risk, discipline, actionability, reasoning, and safety."""

    suite_id = "teacher_benchmark_suite"
    name = "Teacher Model Benchmark Suite"
    version = "v1.0.0"
    description = "Evaluates teacher LLM output quality across coaching structure, risk management, discipline, and safety."
    default_case_ids = [
        "coaching_quality",
        "risk_analysis",
        "discipline_analysis",
        "educational_quality",
        "actionability",
        "reasoning_quality",
        "safety",
    ]

    def run_cases(
        self,
        dataset_artifact: DatasetArtifact,
        enabled_case_ids: list[str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> list[BenchmarkExecutionResult]:
        target_case_ids = enabled_case_ids or self.default_case_ids
        results: list[BenchmarkExecutionResult] = []

        for cid in target_case_ids:
            if cid in CaseRegistry.list_cases():
                case = CaseRegistry.get(cid)
                exec_res = case.evaluate(dataset_artifact, suite_id=self.suite_id, kwargs=kwargs)
                results.append(exec_res)

        return results


@SuiteRegistry.register
class DatasetBenchmarkSuite(BaseBenchmarkSuite):
    """Suite evaluating dataset artifact quality, internal/factual consistency, completeness, and prompt adherence."""

    suite_id = "dataset_benchmark_suite"
    name = "Dataset Quality Benchmark Suite"
    version = "v1.0.0"
    description = (
        "Evaluates dataset integrity, consistency, prompt adherence, completeness, and safety."
    )
    default_case_ids = [
        "internal_consistency",
        "factual_consistency",
        "completeness",
        "prompt_adherence",
        "safety",
    ]

    def run_cases(
        self,
        dataset_artifact: DatasetArtifact,
        enabled_case_ids: list[str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> list[BenchmarkExecutionResult]:
        target_case_ids = enabled_case_ids or self.default_case_ids
        results: list[BenchmarkExecutionResult] = []

        for cid in target_case_ids:
            if cid in CaseRegistry.list_cases():
                case = CaseRegistry.get(cid)
                exec_res = case.evaluate(dataset_artifact, suite_id=self.suite_id, kwargs=kwargs)
                results.append(exec_res)

        return results


@SuiteRegistry.register
class PromptBenchmarkSuite(BaseBenchmarkSuite):
    """Suite evaluating prompt template adherence, instruction following, and formatting compliance."""

    suite_id = "prompt_benchmark_suite"
    name = "Prompt Template Benchmark Suite"
    version = "v1.0.0"
    description = "Evaluates prompt template formatting adherence, instruction follow-through, and response formatting."
    default_case_ids = [
        "prompt_adherence",
        "completeness",
        "actionability",
    ]

    def run_cases(
        self,
        dataset_artifact: DatasetArtifact,
        enabled_case_ids: list[str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> list[BenchmarkExecutionResult]:
        target_case_ids = enabled_case_ids or self.default_case_ids
        results: list[BenchmarkExecutionResult] = []

        for cid in target_case_ids:
            if cid in CaseRegistry.list_cases():
                case = CaseRegistry.get(cid)
                exec_res = case.evaluate(dataset_artifact, suite_id=self.suite_id, kwargs=kwargs)
                results.append(exec_res)

        return results
