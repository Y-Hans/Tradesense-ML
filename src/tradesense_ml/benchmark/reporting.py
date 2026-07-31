"""Benchmark report generator producing independent BenchmarkReport domain objects."""

from typing import Any

from tradesense_ml.domain.schemas.benchmark import (
    BenchmarkMetric,
    BenchmarkProfile,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkScore,
    BenchmarkSummary,
)
from tradesense_ml.domain.schemas.dataset import DatasetArtifact


class BenchmarkReportGenerator:
    """Generates structured BenchmarkReport objects independent of output exporters."""

    @staticmethod
    def generate_report(
        results: list[BenchmarkResult],
        metrics: list[BenchmarkMetric],
        scores: BenchmarkScore,
        summary: BenchmarkSummary,
        profile: BenchmarkProfile,
        dataset_artifact: DatasetArtifact,
        config_dict: dict[str, Any],
        target_model: str = "teacher_llm_v1",
    ) -> BenchmarkReport:
        """Construct independent BenchmarkReport container.

        Args:
            results: List of scored BenchmarkResults.
            metrics: All computed metrics.
            scores: Aggregated BenchmarkScore.
            summary: Executive summary.
            profile: Executed profile.
            dataset_artifact: Input dataset artifact.
            config_dict: Full configuration dictionary.
            target_model: Evaluated model string.

        Returns:
            BenchmarkReport object.
        """
        # Collect metric breakdown
        metric_breakdown = [
            {
                "case_id": res.case_id,
                "case_name": res.case_name,
                "concern": res.concern,
                "score": res.score,
                "weight": res.weight,
                "passed": res.passed,
                "metrics": [
                    {"id": m.metric_id, "name": m.name, "value": m.value, "unit": m.unit}
                    for m in res.metrics
                ],
            }
            for res in results
        ]

        # Failures & warnings
        failures = [
            {
                "case_id": res.case_id,
                "case_name": res.case_name,
                "score": res.score,
                "failure_reasons": res.failure_reasons,
            }
            for res in results
            if not res.passed
        ]

        warnings: list[str] = []
        for res in results:
            warnings.extend(res.warnings)

        # Generate actionable recommendations
        recommendations: list[str] = []
        if summary.overall_score >= 9.0:
            recommendations.append("Model quality is exceptional. Ready for distillation pipeline.")
        elif summary.overall_score >= 7.5:
            recommendations.append(
                "Strong benchmark performance. Fine-tuning recommended to boost weak areas."
            )
        else:
            recommendations.append("Benchmark scores need improvement before model distillation.")

        for res in results:
            if not res.passed:
                recommendations.append(f"Improve performance on '{res.case_name}' ({res.concern}).")

        # Summaries
        config_summary = {
            "profile": profile.profile_id,
            "seed": profile.execution_policy.get("seed", 42),
            "retries": profile.execution_policy.get("retries", 0),
            "config_keys_count": len(config_dict),
        }

        dataset_summary = {
            "dataset_id": dataset_artifact.artifact_id,
            "version": dataset_artifact.dataset_metadata.version,
            "total_examples": dataset_artifact.statistics.total_examples,
            "approved_examples": dataset_artifact.statistics.approved_examples,
            "split_sizes": dataset_artifact.statistics.split_sizes,
        }

        model_summary = {
            "target_model": target_model,
            "prompt_version": dataset_artifact.lineage.prompt_version,
            "synthetic_generator_version": dataset_artifact.lineage.synthetic_generator_version,
        }

        return BenchmarkReport(
            overall_score=summary.overall_score,
            category_scores=summary.category_scores,
            metric_breakdown=metric_breakdown,
            failures=failures,
            warnings=warnings,
            recommendations=recommendations,
            ranking=scores.ranking_info,
            configuration_summary=config_summary,
            dataset_summary=dataset_summary,
            model_summary=model_summary,
        )
