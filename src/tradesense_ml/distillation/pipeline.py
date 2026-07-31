"""Dedicated DistillationPipeline orchestrating strategy execution, statistics, lineage, reporting, validation, and export."""

import hashlib
import json
import time
from typing import Any

from tradesense_ml.distillation.exporters import DistillationExporterManager
from tradesense_ml.distillation.lineage import DistillationLineageTracker
from tradesense_ml.distillation.reporting import DistillationReportGenerator
from tradesense_ml.distillation.runner import DistillationRunner
from tradesense_ml.distillation.statistics import StatisticsGenerator
from tradesense_ml.distillation.validation import DistillationValidator
from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.dataset import DatasetArtifact
from tradesense_ml.domain.schemas.distillation import (
    DistillationArtifact,
    DistillationConfiguration,
    DistillationDataset,
    DistillationManifest,
    DistillationMetadata,
    DistillationSummary,
)
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.pipelines.base import BasePipeline

logger = get_logger()


class DistillationPipeline(BasePipeline[DatasetArtifact, DistillationArtifact]):
    """Orchestrator pipeline consuming canonical DatasetArtifact and optional BenchmarkArtifact to produce canonical DistillationArtifact.

    Architecture Flow:
    DatasetArtifact + BenchmarkArtifact
            ↓
    DistillationPipeline
            ↓
    DistillationRunner
            ↓
    DistillationStrategy (SFTStrategy, DPOStrategy, ORPOStrategy, CurriculumStrategy, HybridStrategy)
            ↓
    Selection -> Filtering -> Sampling -> Curriculum -> Preference
            ↓
    DistillationProcessingResult
            ↓
    Statistics Engine / Lineage Tracker / Report Generator / Validator Engine
            ↓
    DistillationArtifact (Canonical Release)
            ↓
    Exporters (JSON, JSONL, Parquet, Markdown)
    """

    def __init__(
        self,
        runner: DistillationRunner | None = None,
        default_strategy: str = "SFTStrategy",
    ) -> None:
        super().__init__(pipeline_name="distillation_pipeline")
        self.runner = runner or DistillationRunner()
        self.default_strategy = default_strategy

    def run(
        self,
        input_data: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        **kwargs: Any,
    ) -> DistillationArtifact:
        """Run distillation pipeline end-to-end.

        Args:
            input_data: Source DatasetArtifact.
            benchmark_artifact: Optional source BenchmarkArtifact.
            **kwargs: Config overrides (distillation_id, strategy, selection_strategy, sampling_strategy, output_dir, seed).

        Returns:
            Canonical, immutable DistillationArtifact object.
        """
        start_time = time.perf_counter()

        dist_id = str(
            kwargs.get("distillation_id", kwargs.get("artifact_id", "tradesense_distillation_v1"))
        )
        version = str(kwargs.get("version", "v1.0.0"))
        dist_strategy = str(
            kwargs.get("distillation_strategy", kwargs.get("strategy", self.default_strategy))
        )
        sel_strategy = str(kwargs.get("selection_strategy", "ThresholdSelection"))
        samp_strategy = str(kwargs.get("sampling_strategy", "UniformSampling"))
        curr_strategy = str(kwargs.get("curriculum_strategy", "StandardCurriculumStrategy"))
        teacher_model = str(kwargs.get("teacher_model", "teacher_llm_v1"))
        prompt_version = str(kwargs.get("prompt_version", input_data.lineage.prompt_version))
        output_dir = str(kwargs.get("output_dir", "outputs/distillation"))
        export_formats = kwargs.get("export_formats", ["json", "jsonl", "parquet", "md"])
        seed = int(kwargs.get("seed", 42))

        logger.info(
            f"Starting DistillationPipeline for dataset '{input_data.artifact_id}' using strategy '{dist_strategy}'"
        )

        # 1. Pre-execution validation
        pre_val = DistillationValidator.validate_distillation(
            dataset_artifact=input_data, benchmark_artifact=benchmark_artifact
        )
        if not pre_val.is_valid:
            error_msg = f"Pre-execution validation failed: {'; '.join(pre_val.errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        config_dict = {
            "distillation_id": dist_id,
            "version": version,
            "distillation_strategy": dist_strategy,
            "selection_strategy": sel_strategy,
            "sampling_strategy": samp_strategy,
            "curriculum_strategy": curr_strategy,
            "teacher_model": teacher_model,
            "random_seed": seed,
            "kwargs": kwargs,
        }

        # 2. Delegate execution to DistillationRunner -> DistillationProcessingResult
        processing_result = self.runner.run_strategy(
            dataset_artifact=input_data,
            benchmark_artifact=benchmark_artifact,
            strategy_name=dist_strategy,
            config_dict=config_dict,
            **kwargs,
        )

        # 3. Generate statistics from processing result
        statistics = StatisticsGenerator.generate_statistics(processing_result=processing_result)

        # 4. Generate lineage & provenance tracking
        lineage = DistillationLineageTracker.create_lineage(
            dataset_artifact=input_data,
            benchmark_artifact=benchmark_artifact,
            config_dict=config_dict,
            distillation_strategy=dist_strategy,
            selection_strategy=sel_strategy,
            sampling_strategy=samp_strategy,
            curriculum_strategy=curr_strategy,
            teacher_model=teacher_model,
            prompt_version=prompt_version,
            random_seed=seed,
        )

        metadata = DistillationMetadata(
            artifact_id=dist_id,
            name=f"Distillation Release - {dist_strategy}",
            description=f"Distillation dataset release prepared from {input_data.artifact_id}",
            version=version,
            dataset_artifact_id=input_data.artifact_id,
            benchmark_artifact_id=benchmark_artifact.artifact_id if benchmark_artifact else "none",
            teacher_model=teacher_model,
            prompt_version=prompt_version,
        )

        exec_time = time.perf_counter() - start_time
        overall_mean = (
            sum(e.quality_score for e in processing_result.sampled_examples)
            / max(1, len(processing_result.sampled_examples))
            if processing_result.sampled_examples
            else 0.0
        )

        summary = DistillationSummary(
            artifact_id=dist_id,
            total_input_examples=input_data.statistics.total_examples,
            total_selected_examples=len(processing_result.selected_examples),
            total_sampled_examples=len(processing_result.sampled_examples),
            total_preference_pairs=len(processing_result.preference_pairs),
            total_curriculum_stages=len(processing_result.curriculum_stages),
            overall_quality_mean=overall_mean,
            execution_time_seconds=exec_time,
        )

        # 5. Generate independent DistillationReport
        report = DistillationReportGenerator.generate_report(
            processing_result=processing_result,
            statistics=statistics,
            dataset_artifact=input_data,
            benchmark_artifact=benchmark_artifact,
            config_dict=config_dict,
            teacher_model=teacher_model,
        )

        config_model = DistillationConfiguration(
            distillation_strategy=dist_strategy,
            selection_strategy=sel_strategy,
            sampling_strategy=samp_strategy,
            curriculum_strategy=curr_strategy,
            export_formats=export_formats,
            random_seed=seed,
            output_dir=output_dir,
        )

        dist_dataset = DistillationDataset(
            sft_examples=processing_result.sampled_examples,
            preference_pairs=processing_result.preference_pairs,
            curriculum_stages=processing_result.curriculum_stages,
            total_examples=len(processing_result.sampled_examples),
            total_preference_pairs=len(processing_result.preference_pairs),
        )

        manifest_data = {
            "artifact_id": dist_id,
            "version": version,
            "total_examples": len(processing_result.sampled_examples),
            "configuration_hash": lineage.configuration_hash,
        }
        manifest_chk = hashlib.sha256(
            json.dumps(manifest_data, sort_keys=True).encode()
        ).hexdigest()

        manifest = DistillationManifest(
            artifact_id=dist_id,
            version=version,
            statistics_summary=statistics.selection_counts,
            configuration_hash=lineage.configuration_hash,
            lineage={"dataset_id": input_data.artifact_id},
            export_files=[],
            checksum=manifest_chk,
        )

        bm_ref = {}
        if benchmark_artifact is not None:
            bm_ref = {
                "artifact_id": benchmark_artifact.artifact_id,
                "overall_score": benchmark_artifact.scores.overall_score,
            }

        # 6. Provisional DistillationArtifact
        artifact = DistillationArtifact(
            artifact_id=dist_id,
            metadata=metadata,
            lineage=lineage,
            configuration=config_model,
            summary=summary,
            statistics=statistics,
            manifest=manifest,
            dataset=dist_dataset,
            report=report,
            export_files=[],
            dataset_reference={"artifact_id": input_data.artifact_id},
            benchmark_reference=bm_ref,
        )

        # 7. Post-execution validation
        post_val = DistillationValidator.validate_distillation(
            dataset_artifact=input_data,
            benchmark_artifact=benchmark_artifact,
            artifact=artifact,
        )
        if not post_val.is_valid:
            error_msg = f"Post-execution validation failed: {'; '.join(post_val.errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 8. Export DistillationArtifact via Exporters
        descriptors = DistillationExporterManager.export_artifact(
            artifact=artifact,
            output_dir=output_dir,
            formats=export_formats,
        )

        # 9. Return final immutable DistillationArtifact
        final_manifest = DistillationManifest(
            artifact_id=dist_id,
            version=version,
            statistics_summary=statistics.selection_counts,
            configuration_hash=lineage.configuration_hash,
            lineage={"dataset_id": input_data.artifact_id},
            export_files=descriptors,
            checksum=manifest_chk,
        )

        final_artifact = DistillationArtifact(
            artifact_id=dist_id,
            metadata=metadata,
            lineage=lineage,
            configuration=config_model,
            summary=summary,
            statistics=statistics,
            manifest=final_manifest,
            dataset=dist_dataset,
            report=report,
            export_files=descriptors,
            dataset_reference={"artifact_id": input_data.artifact_id},
            benchmark_reference=bm_ref,
        )

        logger.info(
            f"Successfully completed DistillationPipeline in {exec_time:.2f}s. Artifact ID: '{dist_id}'"
        )
        return final_artifact
