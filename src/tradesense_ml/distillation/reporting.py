"""Distillation Report Generator creating independent DistillationReport containers."""

from typing import Any

from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.dataset import DatasetArtifact
from tradesense_ml.domain.schemas.distillation import (
    DistillationProcessingResult,
    DistillationReport,
    DistillationStatistics,
)


class DistillationReportGenerator:
    """Generator creating independent DistillationReport objects."""

    @staticmethod
    def generate_report(
        processing_result: DistillationProcessingResult,
        statistics: DistillationStatistics,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        config_dict: dict[str, Any] | None = None,
        teacher_model: str = "teacher_llm_v1",
    ) -> DistillationReport:
        cfg = config_dict or {}

        # 1. Summaries
        sel_sum = {
            "strategy": processing_result.selection_result.strategy_name,
            "threshold_applied": processing_result.selection_result.threshold_applied,
            "selected_count": len(processing_result.selected_examples),
        }
        filt_sum = processing_result.filtering_stats
        samp_sum = {
            "strategy": processing_result.sampling_result.strategy_name,
            "sample_size": processing_result.sampling_result.sample_size,
            "sampling_rate": processing_result.sampling_result.sampling_rate,
        }
        curr_sum = {
            "stages_count": len(processing_result.curriculum_stages),
            "distribution": statistics.curriculum_distribution,
        }
        pref_sum = {
            "pairs_count": len(processing_result.preference_pairs),
        }

        # 2. Warnings & Recommendations
        warnings: list[str] = []
        recommendations: list[str] = []

        if len(processing_result.sampled_examples) == 0:
            warnings.append("No examples were selected or sampled for distillation.")
            recommendations.append("Lower selection quality threshold or adjust filtering rules.")

        if statistics.dataset_size_bytes < 1000:
            warnings.append("Dataset size is small (< 1 KB).")

        if len(processing_result.preference_pairs) > 0:
            recommendations.append(
                "Preference dataset ready for direct consumption by DPO/ORPO fine-tuning pipelines."
            )
        else:
            recommendations.append(
                "Consider running DPOStrategy or HybridStrategy if preference alignment is required."
            )

        recommendations.append(
            "Ensure canonical DistillationArtifact is stored with version manifest prior to fine-tuning."
        )

        # 3. Input summaries
        ds_sum = {
            "artifact_id": dataset_artifact.artifact_id,
            "version": dataset_artifact.dataset_metadata.version,
            "total_examples": dataset_artifact.statistics.total_examples,
        }

        bm_sum = {}
        if benchmark_artifact is not None:
            bm_sum = {
                "artifact_id": benchmark_artifact.artifact_id,
                "overall_score": benchmark_artifact.scores.overall_score,
                "pass_rate": benchmark_artifact.summary.pass_rate,
            }

        return DistillationReport(
            selection_summary=sel_sum,
            filtering_summary=filt_sum,
            sampling_summary=samp_sum,
            curriculum_summary=curr_sum,
            preference_summary=pref_sum,
            statistics=statistics,
            warnings=warnings,
            recommendations=recommendations,
            configuration_summary=cfg,
            dataset_summary=ds_sum,
            benchmark_summary=bm_sum,
        )
