"""Filtering Engine for cleaning and sanitizing distillation examples."""

from typing import Any

from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.distillation import DistillationExample
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class FilteringEngine:
    """Independent Filtering Engine applying strict sanitation, deduplication, and quality criteria."""

    def filter_examples(
        self,
        examples: list[DistillationExample],
        benchmark_artifact: BenchmarkArtifact | None = None,
        min_quality_score: float = 6.0,
        remove_duplicates: bool = True,
        remove_incomplete: bool = True,
        remove_malformed: bool = True,
        remove_unsafe: bool = True,
        min_output_length: int = 10,
        **kwargs: Any,
    ) -> tuple[list[DistillationExample], list[DistillationExample], dict[str, Any]]:
        """Filter examples and return (passed_examples, rejected_examples, stats_dict).

        Returns:
            passed_examples: List of examples meeting all criteria.
            rejected_examples: List of rejected examples.
            stats_dict: Breakdown of rejection reasons and pass counts.
        """
        passed: list[DistillationExample] = []
        rejected: list[DistillationExample] = []

        seen_inputs: set[str] = set()
        seen_ids: set[str] = set()

        rejection_reasons: dict[str, int] = {
            "duplicate": 0,
            "low_quality": 0,
            "incomplete": 0,
            "malformed": 0,
            "unsafe": 0,
            "empty": 0,
            "benchmark_failure": 0,
        }

        # Build set of failed benchmark case target keywords if available
        failed_cases: set[str] = set()
        if benchmark_artifact is not None:
            for r in benchmark_artifact.results:
                if not r.passed:
                    failed_cases.add(r.case_id.lower())

        for ex in examples:
            is_rejected = False
            reason = ""

            # 1. Empty check
            if not ex.input.strip() and not ex.output.strip():
                is_rejected = True
                reason = "empty"

            # 2. Incomplete check
            elif remove_incomplete and (
                not ex.instruction.strip() or not ex.input.strip() or not ex.output.strip()
            ):
                is_rejected = True
                reason = "incomplete"

            # 3. Malformed / Minimum length check
            elif remove_malformed and len(ex.output.strip()) < min_output_length:
                is_rejected = True
                reason = "malformed"

            # 4. Low-quality check
            elif ex.quality_score < min_quality_score:
                is_rejected = True
                reason = "low_quality"

            # 5. Duplicate check
            elif remove_duplicates:
                input_norm = ex.input.strip().lower()
                if ex.example_id in seen_ids or input_norm in seen_inputs:
                    is_rejected = True
                    reason = "duplicate"
                else:
                    seen_ids.add(ex.example_id)
                    seen_inputs.add(input_norm)

            # 6. Unsafe output check
            elif remove_unsafe and (
                ex.metadata.get("unsafe", False)
                or "guaranteed 100% profit" in ex.output.lower()
                or "insider info" in ex.output.lower()
            ):
                is_rejected = True
                reason = "unsafe"

            # 7. Benchmark failure check
            elif benchmark_artifact is not None and any(
                fc in ex.instruction.lower() for fc in failed_cases
            ):
                is_rejected = True
                reason = "benchmark_failure"

            if is_rejected:
                rejected.append(ex)
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            else:
                passed.append(ex)

        stats = {
            "total_input": len(examples),
            "passed_count": len(passed),
            "rejected_count": len(rejected),
            "rejection_reasons": rejection_reasons,
        }

        logger.info(
            f"FilteringEngine completed: {len(passed)}/{len(examples)} passed, {len(rejected)} rejected."
        )
        return passed, rejected, stats
