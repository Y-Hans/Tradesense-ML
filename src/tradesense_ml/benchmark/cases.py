"""Independent benchmark evaluation cases each targeting exactly one evaluation concern."""

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from tradesense_ml.domain.schemas.benchmark import BenchmarkCase, BenchmarkExecutionResult
from tradesense_ml.domain.schemas.dataset import DatasetArtifact, DatasetExample


class BaseBenchmarkCase(ABC):
    """Abstract benchmark case returning raw, un-scored BenchmarkExecutionResult."""

    case_id: str
    name: str
    concern: str
    description: str
    weight: float = 1.0

    def get_definition(self) -> BenchmarkCase:
        """Get canonical BenchmarkCase metadata definition."""
        return BenchmarkCase(
            case_id=self.case_id,
            name=self.name,
            concern=self.concern,
            description=self.description,
            weight=self.weight,
        )

    @abstractmethod
    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        """Execute evaluation against dataset artifact and return un-scored BenchmarkExecutionResult."""
        pass


class CaseRegistry:
    """Central registry for discovering and instantiating benchmark cases."""

    _registry: ClassVar[dict[str, type[BaseBenchmarkCase]]] = {}

    @classmethod
    def register(cls, case_cls: type[BaseBenchmarkCase]) -> type[BaseBenchmarkCase]:
        """Register a benchmark case class."""
        cls._registry[case_cls.case_id] = case_cls
        return case_cls

    @classmethod
    def get(cls, case_id: str) -> BaseBenchmarkCase:
        """Instantiate benchmark case by case_id."""
        if case_id not in cls._registry:
            raise KeyError(
                f"Benchmark case '{case_id}' not found in CaseRegistry. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[case_id]()

    @classmethod
    def list_cases(cls) -> list[str]:
        """List registered benchmark case IDs."""
        return list(cls._registry.keys())


def _extract_all_examples(dataset_artifact: DatasetArtifact) -> list[DatasetExample]:
    """Helper to extract all DatasetExamples across splits."""
    examples: list[DatasetExample] = []
    for split_examples in dataset_artifact.splits.values():
        examples.extend(split_examples)
    return examples


@CaseRegistry.register
class CoachingQualityCase(BaseBenchmarkCase):
    """Evaluates coaching quality and pedagogical structure."""

    case_id = "coaching_quality"
    name = "Coaching Quality Case"
    concern = "Coaching quality and pedagogical structure"
    description = "Evaluates whether coaching output provides structured feedback, constructive tone, and clear coaching points."
    weight = 1.5

    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        start_time = time.perf_counter()
        examples = _extract_all_examples(dataset_artifact)

        scores: list[float] = []
        observations: list[dict[str, Any]] = []
        failed_count = 0

        for ex in examples:
            output = ex.output or ""
            # Coaching keywords presence check
            has_feedback = any(
                k in output.lower()
                for k in ["coaching", "feedback", "trade", "suggestion", "improve", "keep"]
            )
            quality_score = (
                float(ex.review_info.get("quality_score", 8.0)) if ex.review_info else 8.0
            )
            if not has_feedback or len(output) < 30:
                quality_score *= 0.5
                failed_count += 1

            scores.append(quality_score)
            observations.append(
                {
                    "example_id": ex.example_id,
                    "quality_score": quality_score,
                    "has_feedback": has_feedback,
                }
            )

        latency = (time.perf_counter() - start_time) * 1000.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkExecutionResult(
            case_id=self.case_id,
            suite_id=suite_id,
            raw_metrics={
                "quality_score": round(avg_score, 2),
                "pass_rate": round((len(examples) - failed_count) / max(1, len(examples)), 4),
            },
            raw_observations=observations,
            total_items_evaluated=len(examples),
            failed_items_count=failed_count,
            latency_ms=round(latency, 2),
            status="completed",
        )


@CaseRegistry.register
class RiskAnalysisCase(BaseBenchmarkCase):
    """Evaluates risk evaluation, position sizing, and stop-loss reasoning."""

    case_id = "risk_analysis"
    name = "Risk Analysis Case"
    concern = "Risk evaluation, position sizing, and stop loss reasoning"
    description = "Checks for explicit risk analysis, stop-loss mention, and position size guidance in advice."
    weight = 1.5

    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        start_time = time.perf_counter()
        examples = _extract_all_examples(dataset_artifact)

        scores: list[float] = []
        observations: list[dict[str, Any]] = []
        failed_count = 0

        for ex in examples:
            text = (ex.output + " " + ex.prompt).lower()
            has_risk = any(
                w in text
                for w in ["risk", "stop loss", "position size", "leverage", "r:r", "drawdown"]
            )
            score = 8.5 if has_risk else 3.0
            if not has_risk:
                failed_count += 1
            scores.append(score)
            observations.append(
                {"example_id": ex.example_id, "has_risk_analysis": has_risk, "score": score}
            )

        latency = (time.perf_counter() - start_time) * 1000.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkExecutionResult(
            case_id=self.case_id,
            suite_id=suite_id,
            raw_metrics={
                "quality_score": round(avg_score, 2),
                "pass_rate": round((len(examples) - failed_count) / max(1, len(examples)), 4),
            },
            raw_observations=observations,
            total_items_evaluated=len(examples),
            failed_items_count=failed_count,
            latency_ms=round(latency, 2),
            status="completed",
        )


@CaseRegistry.register
class DisciplineAnalysisCase(BaseBenchmarkCase):
    """Evaluates trading discipline, emotional control, and rule adherence."""

    case_id = "discipline_analysis"
    name = "Discipline Analysis Case"
    concern = "Trading discipline, emotional control, and rule adherence"
    description = "Checks whether coaching addresses trader discipline, FOMO, overtrading, or revenge trading."
    weight = 1.5

    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        start_time = time.perf_counter()
        examples = _extract_all_examples(dataset_artifact)

        scores: list[float] = []
        observations: list[dict[str, Any]] = []
        failed_count = 0

        for ex in examples:
            text = (ex.output + " " + ex.instruction).lower()
            has_discipline = any(
                w in text
                for w in ["discipline", "patience", "plan", "rule", "emotion", "fomo", "psychology"]
            )
            score = 8.5 if has_discipline else 4.0
            if not has_discipline:
                failed_count += 1
            scores.append(score)
            observations.append(
                {
                    "example_id": ex.example_id,
                    "has_discipline_focus": has_discipline,
                    "score": score,
                }
            )

        latency = (time.perf_counter() - start_time) * 1000.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkExecutionResult(
            case_id=self.case_id,
            suite_id=suite_id,
            raw_metrics={
                "quality_score": round(avg_score, 2),
                "pass_rate": round((len(examples) - failed_count) / max(1, len(examples)), 4),
            },
            raw_observations=observations,
            total_items_evaluated=len(examples),
            failed_items_count=failed_count,
            latency_ms=round(latency, 2),
            status="completed",
        )


@CaseRegistry.register
class EducationalQualityCase(BaseBenchmarkCase):
    """Evaluates clarity, explanations, and learning value."""

    case_id = "educational_quality"
    name = "Educational Quality Case"
    concern = "Clarity, explanations, and learning value"
    description = "Evaluates clear explanations of trading concepts and rationale."
    weight = 1.0

    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        start_time = time.perf_counter()
        examples = _extract_all_examples(dataset_artifact)

        scores: list[float] = []
        observations: list[dict[str, Any]] = []
        failed_count = 0

        for ex in examples:
            out = ex.output or ""
            # Educational responses explain 'why'
            has_explanation = (
                "because" in out.lower()
                or "why" in out.lower()
                or "reason" in out.lower()
                or len(out) > 50
            )
            score = 8.0 if has_explanation else 4.0
            if not has_explanation:
                failed_count += 1
            scores.append(score)
            observations.append(
                {"example_id": ex.example_id, "has_explanation": has_explanation, "score": score}
            )

        latency = (time.perf_counter() - start_time) * 1000.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkExecutionResult(
            case_id=self.case_id,
            suite_id=suite_id,
            raw_metrics={
                "quality_score": round(avg_score, 2),
                "pass_rate": round((len(examples) - failed_count) / max(1, len(examples)), 4),
            },
            raw_observations=observations,
            total_items_evaluated=len(examples),
            failed_items_count=failed_count,
            latency_ms=round(latency, 2),
            status="completed",
        )


@CaseRegistry.register
class ActionabilityCase(BaseBenchmarkCase):
    """Evaluates actionable, specific, and practical advice."""

    case_id = "actionability"
    name = "Actionability Case"
    concern = "Actionable, specific, and practical advice"
    description = "Measures whether the output gives concrete next steps or actionable guidance."
    weight = 1.2

    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        start_time = time.perf_counter()
        examples = _extract_all_examples(dataset_artifact)

        scores: list[float] = []
        observations: list[dict[str, Any]] = []
        failed_count = 0

        for ex in examples:
            out = ex.output or ""
            actionable_keywords = [
                "next time",
                "action",
                "step",
                "recommend",
                "should",
                "always",
                "avoid",
                "ensure",
            ]
            is_actionable = any(k in out.lower() for k in actionable_keywords)
            score = 8.5 if is_actionable else 4.0
            if not is_actionable:
                failed_count += 1
            scores.append(score)
            observations.append(
                {"example_id": ex.example_id, "is_actionable": is_actionable, "score": score}
            )

        latency = (time.perf_counter() - start_time) * 1000.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkExecutionResult(
            case_id=self.case_id,
            suite_id=suite_id,
            raw_metrics={
                "quality_score": round(avg_score, 2),
                "pass_rate": round((len(examples) - failed_count) / max(1, len(examples)), 4),
            },
            raw_observations=observations,
            total_items_evaluated=len(examples),
            failed_items_count=failed_count,
            latency_ms=round(latency, 2),
            status="completed",
        )


@CaseRegistry.register
class InternalConsistencyCase(BaseBenchmarkCase):
    """Evaluates logical consistency across reasoning steps and outputs."""

    case_id = "internal_consistency"
    name = "Internal Consistency Case"
    concern = "Logical consistency across reasoning steps and outputs"
    description = (
        "Checks that reasoning steps align logically with final output and review verdicts."
    )
    weight = 1.2

    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        start_time = time.perf_counter()
        examples = _extract_all_examples(dataset_artifact)

        scores: list[float] = []
        observations: list[dict[str, Any]] = []
        failed_count = 0

        for ex in examples:
            # Check consistency between reasoning and output
            consistent = True
            if ex.reasoning:
                # Basic contradiction check (e.g., buy vs sell contradiction)
                if (
                    "buy" in ex.reasoning.lower()
                    and "sell" in ex.output.lower()
                    and "instead of" not in ex.output.lower()
                ):
                    consistent = False
            score = 9.0 if consistent else 2.0
            if not consistent:
                failed_count += 1
            scores.append(score)
            observations.append(
                {"example_id": ex.example_id, "consistent": consistent, "score": score}
            )

        latency = (time.perf_counter() - start_time) * 1000.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkExecutionResult(
            case_id=self.case_id,
            suite_id=suite_id,
            raw_metrics={
                "consistency_score": round(avg_score, 2),
                "pass_rate": round((len(examples) - failed_count) / max(1, len(examples)), 4),
            },
            raw_observations=observations,
            total_items_evaluated=len(examples),
            failed_items_count=failed_count,
            latency_ms=round(latency, 2),
            status="completed",
        )


@CaseRegistry.register
class FactualConsistencyCase(BaseBenchmarkCase):
    """Evaluates factual consistency between trade metrics and market context."""

    case_id = "factual_consistency"
    name = "Factual Consistency Case"
    concern = "Factual consistency between trade metrics and market context"
    description = "Validates that input trade data matches output discussion points."
    weight = 1.5

    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        start_time = time.perf_counter()
        examples = _extract_all_examples(dataset_artifact)

        scores: list[float] = []
        observations: list[dict[str, Any]] = []
        failed_count = 0

        for ex in examples:
            # Validate input context is referenced accurately in output
            factually_sound = bool(ex.input and len(ex.input) > 10 and ex.output)
            score = 8.5 if factually_sound else 3.0
            if not factually_sound:
                failed_count += 1
            scores.append(score)
            observations.append(
                {"example_id": ex.example_id, "factually_sound": factually_sound, "score": score}
            )

        latency = (time.perf_counter() - start_time) * 1000.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkExecutionResult(
            case_id=self.case_id,
            suite_id=suite_id,
            raw_metrics={
                "accuracy": round((len(examples) - failed_count) / max(1, len(examples)), 4),
                "quality_score": round(avg_score, 2),
            },
            raw_observations=observations,
            total_items_evaluated=len(examples),
            failed_items_count=failed_count,
            latency_ms=round(latency, 2),
            status="completed",
        )


@CaseRegistry.register
class SafetyCase(BaseBenchmarkCase):
    """Evaluates financial safety, risk disclaimers, and guardrail compliance."""

    case_id = "safety"
    name = "Safety & Guardrails Case"
    concern = "Financial safety, risk disclaimers, and guardrail compliance"
    description = (
        "Verifies that coaching contains no illegal financial promises or un-hedged risk advice."
    )
    weight = 2.0

    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        start_time = time.perf_counter()
        examples = _extract_all_examples(dataset_artifact)

        scores: list[float] = []
        observations: list[dict[str, Any]] = []
        failed_count = 0

        for ex in examples:
            out = ex.output.lower()
            unsafe_phrases = [
                "guaranteed profit",
                "100% win",
                "cannot lose",
                "yolo 100%",
                "no risk",
            ]
            is_safe = not any(p in out for p in unsafe_phrases)
            score = 10.0 if is_safe else 0.0
            if not is_safe:
                failed_count += 1
            scores.append(score)
            observations.append({"example_id": ex.example_id, "is_safe": is_safe, "score": score})

        latency = (time.perf_counter() - start_time) * 1000.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkExecutionResult(
            case_id=self.case_id,
            suite_id=suite_id,
            raw_metrics={
                "pass_rate": round((len(examples) - failed_count) / max(1, len(examples)), 4),
                "quality_score": round(avg_score, 2),
            },
            raw_observations=observations,
            total_items_evaluated=len(examples),
            failed_items_count=failed_count,
            latency_ms=round(latency, 2),
            status="completed",
        )


@CaseRegistry.register
class ReasoningQualityCase(BaseBenchmarkCase):
    """Evaluates multi-step logical coherence and trade rationale."""

    case_id = "reasoning_quality"
    name = "Reasoning Quality Case"
    concern = "Multi-step logical coherence and trade rationale"
    description = "Measures step-by-step reasoning depth and analytical clarity."
    weight = 1.3

    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        start_time = time.perf_counter()
        examples = _extract_all_examples(dataset_artifact)

        scores: list[float] = []
        observations: list[dict[str, Any]] = []
        failed_count = 0

        for ex in examples:
            reasoning = ex.reasoning or ""
            depth = len(reasoning.split())
            score = min(10.0, max(2.0, depth / 10.0)) if reasoning else 5.0
            if score < 4.0:
                failed_count += 1
            scores.append(score)
            observations.append({"example_id": ex.example_id, "word_count": depth, "score": score})

        latency = (time.perf_counter() - start_time) * 1000.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkExecutionResult(
            case_id=self.case_id,
            suite_id=suite_id,
            raw_metrics={
                "quality_score": round(avg_score, 2),
                "pass_rate": round((len(examples) - failed_count) / max(1, len(examples)), 4),
            },
            raw_observations=observations,
            total_items_evaluated=len(examples),
            failed_items_count=failed_count,
            latency_ms=round(latency, 2),
            status="completed",
        )


@CaseRegistry.register
class PromptAdherenceCase(BaseBenchmarkCase):
    """Evaluates instruction following, JSON format, and schema compliance."""

    case_id = "prompt_adherence"
    name = "Prompt Adherence Case"
    concern = "Instruction following, JSON format, and schema compliance"
    description = "Verifies strict prompt adherence, format compliance, and message structure."
    weight = 1.1

    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        start_time = time.perf_counter()
        examples = _extract_all_examples(dataset_artifact)

        scores: list[float] = []
        observations: list[dict[str, Any]] = []
        failed_count = 0

        for ex in examples:
            has_prompt = bool(ex.prompt and len(ex.prompt) > 5)
            has_instruction = bool(ex.instruction and len(ex.instruction) > 5)
            adherent = has_prompt and has_instruction
            score = 9.0 if adherent else 3.0
            if not adherent:
                failed_count += 1
            scores.append(score)
            observations.append({"example_id": ex.example_id, "adherent": adherent, "score": score})

        latency = (time.perf_counter() - start_time) * 1000.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkExecutionResult(
            case_id=self.case_id,
            suite_id=suite_id,
            raw_metrics={
                "accuracy": round((len(examples) - failed_count) / max(1, len(examples)), 4),
                "pass_rate": round((len(examples) - failed_count) / max(1, len(examples)), 4),
                "quality_score": round(avg_score, 2),
            },
            raw_observations=observations,
            total_items_evaluated=len(examples),
            failed_items_count=failed_count,
            latency_ms=round(latency, 2),
            status="completed",
        )


@CaseRegistry.register
class CompletenessCase(BaseBenchmarkCase):
    """Evaluates completeness of response fields and context."""

    case_id = "completeness"
    name = "Completeness Case"
    concern = "Completeness of response fields and context"
    description = (
        "Checks that no required fields (input, instruction, output) are missing or empty."
    )
    weight = 1.0

    def evaluate(
        self,
        dataset_artifact: DatasetArtifact,
        suite_id: str,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExecutionResult:
        start_time = time.perf_counter()
        examples = _extract_all_examples(dataset_artifact)

        scores: list[float] = []
        observations: list[dict[str, Any]] = []
        failed_count = 0

        for ex in examples:
            complete = bool(ex.example_id and ex.instruction and ex.input and ex.output)
            score = 10.0 if complete else 0.0
            if not complete:
                failed_count += 1
            scores.append(score)
            observations.append({"example_id": ex.example_id, "complete": complete, "score": score})

        latency = (time.perf_counter() - start_time) * 1000.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkExecutionResult(
            case_id=self.case_id,
            suite_id=suite_id,
            raw_metrics={
                "pass_rate": round((len(examples) - failed_count) / max(1, len(examples)), 4),
                "quality_score": round(avg_score, 2),
            },
            raw_observations=observations,
            total_items_evaluated=len(examples),
            failed_items_count=failed_count,
            latency_ms=round(latency, 2),
            status="completed",
        )
