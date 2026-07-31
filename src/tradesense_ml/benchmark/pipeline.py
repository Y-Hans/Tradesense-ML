"""Dedicated BenchmarkPipeline orchestrating benchmark execution, scoring, lineage, reporting, validation, and export."""

import time
from typing import Any

from tradesense_ml.benchmark.exporters import BenchmarkExporterManager
from tradesense_ml.benchmark.lineage import BenchmarkLineageTracker
from tradesense_ml.benchmark.profiles import ProfileRegistry
from tradesense_ml.benchmark.reporting import BenchmarkReportGenerator
from tradesense_ml.benchmark.runner import BenchmarkRunner
from tradesense_ml.benchmark.scoring import BenchmarkScoringEngine
from tradesense_ml.benchmark.validation import BenchmarkValidator
from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact, BenchmarkMetadata
from tradesense_ml.domain.schemas.dataset import DatasetArtifact
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.pipelines.base import BasePipeline

logger = get_logger()


class BenchmarkPipeline(BasePipeline[DatasetArtifact, BenchmarkArtifact]):
    """Orchestrator pipeline consuming canonical DatasetArtifact and producing canonical BenchmarkArtifact.

    Architecture Flow:
    DatasetArtifact
           ↓
    BenchmarkProfile
           ↓
    BenchmarkPipeline
           ↓
    BenchmarkRunner (Executes BenchmarkSuite -> BenchmarkCase)
           ↓
    BenchmarkExecutionResult (Raw Measurements)
           ↓
    BenchmarkScoringEngine (Evaluates standardized BenchmarkResult & BenchmarkScore)
           ↓
    Lineage & Report Generation
           ↓
    BenchmarkArtifact (Canonical Release)
    """

    def __init__(
        self,
        runner: BenchmarkRunner | None = None,
        scoring_engine: BenchmarkScoringEngine | None = None,
        profile_name: str = "teacher_evaluation",
    ) -> None:
        super().__init__(pipeline_name="benchmark_pipeline")
        self.runner = runner or BenchmarkRunner()
        self.scoring_engine = scoring_engine or BenchmarkScoringEngine()
        self.profile_name = profile_name

    def run(self, input_data: DatasetArtifact, **kwargs: Any) -> BenchmarkArtifact:
        """Run benchmark pipeline end-to-end.

        Args:
            input_data: Canonical DatasetArtifact instance.
            **kwargs: Configuration overrides (profile, benchmark_id, target_model, output_dir, export_formats, seed).

        Returns:
            Canonical, immutable BenchmarkArtifact object.
        """
        start_time = time.perf_counter()

        profile_id = str(kwargs.get("profile", kwargs.get("profile_name", self.profile_name)))
        profile = ProfileRegistry.get(profile_id)

        bm_id = str(kwargs.get("benchmark_id", f"benchmark_{profile_id}_v1"))
        version = str(kwargs.get("version", "v1.0.0"))
        target_model = str(kwargs.get("target_model", "teacher_llm_v1"))
        student_model = kwargs.get("student_model")
        prompt_version = str(kwargs.get("prompt_version", input_data.lineage.prompt_version))
        output_dir = str(kwargs.get("output_dir", "outputs/benchmarks"))
        export_formats = kwargs.get("export_formats", ["json", "jsonl", "parquet", "md"])
        seed = int(kwargs.get("seed", 42))

        logger.info(
            f"Starting BenchmarkPipeline for dataset '{input_data.artifact_id}' using profile '{profile.name}' (target_model={target_model})"
        )

        # 1. Pre-execution validation
        pre_val = BenchmarkValidator.validate_benchmark(
            dataset_artifact=input_data, profile=profile
        )
        if not pre_val.is_valid:
            error_msg = (
                f"BenchmarkPipeline pre-execution validation failed: {'; '.join(pre_val.errors)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 2. Execute suites/cases via BenchmarkRunner -> Collect raw BenchmarkExecutionResult objects
        execution_results = self.runner.run_profile(
            dataset_artifact=input_data,
            profile=profile,
            kwargs=kwargs,
        )

        # 3. Evaluate raw measurements via BenchmarkScoringEngine -> BenchmarkResult, metrics, BenchmarkScore, BenchmarkSummary
        results, metrics, scores, summary = self.scoring_engine.evaluate(
            execution_results=execution_results,
            profile=profile,
            pass_threshold=kwargs.get("pass_threshold", 6.0),
        )

        # 4. Generate lineage & provenance tracking
        config_dict = {
            "benchmark_id": bm_id,
            "version": version,
            "profile_id": profile_id,
            "target_model": target_model,
            "seed": seed,
            "kwargs": kwargs,
        }

        lineage = BenchmarkLineageTracker.create_lineage(
            dataset_artifact=input_data,
            config_dict=config_dict,
            teacher_model=target_model,
            student_model=student_model,
            prompt_version=prompt_version,
            benchmark_version=version,
            random_seed=seed,
        )

        metadata = BenchmarkMetadata(
            benchmark_id=bm_id,
            name=f"Benchmark Run - {profile.name}",
            description=f"Evaluation of {target_model} against DatasetArtifact '{input_data.artifact_id}'",
            version=version,
            suite_name=",".join(profile.suite_names),
            profile_name=profile.profile_id,
            target_model=target_model,
            student_model=student_model,
            prompt_version=prompt_version,
            dataset_id=input_data.artifact_id,
            dataset_version=input_data.dataset_metadata.version,
        )

        # 5. Generate independent BenchmarkReport
        report = BenchmarkReportGenerator.generate_report(
            results=results,
            metrics=metrics,
            scores=scores,
            summary=summary,
            profile=profile,
            dataset_artifact=input_data,
            config_dict=config_dict,
            target_model=target_model,
        )

        # 6. Create provisional BenchmarkArtifact
        artifact = BenchmarkArtifact(
            artifact_id=bm_id,
            metadata=metadata,
            lineage=lineage,
            profile=profile,
            suite_info={
                "suites": profile.suite_names,
                "case_count": len(results),
                "pass_rate": summary.pass_rate,
            },
            execution_results=execution_results,
            results=results,
            metrics=metrics,
            scores=scores,
            summary=summary,
            report=report,
            configuration=config_dict,
            dataset_reference={
                "artifact_id": input_data.artifact_id,
                "version": input_data.dataset_metadata.version,
                "examples_count": input_data.statistics.total_examples,
            },
            model_reference={
                "teacher_model": target_model,
                "student_model": student_model,
            },
            prompt_reference={
                "prompt_version": prompt_version,
            },
            export_files=[],
        )

        # 7. Post-execution validation check
        post_val = BenchmarkValidator.validate_benchmark(
            dataset_artifact=input_data,
            profile=profile,
            artifact=artifact,
        )
        if not post_val.is_valid:
            error_msg = (
                f"BenchmarkArtifact post-execution validation failed: {'; '.join(post_val.errors)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 8. Export BenchmarkArtifact via BenchmarkExporterManager
        file_descriptors = BenchmarkExporterManager.export_artifact(
            artifact=artifact,
            output_dir=output_dir,
            formats=export_formats,
        )

        # 9. Return final immutable BenchmarkArtifact
        final_artifact = BenchmarkArtifact(
            artifact_id=bm_id,
            metadata=metadata,
            lineage=lineage,
            profile=profile,
            suite_info=artifact.suite_info,
            execution_results=execution_results,
            results=results,
            metrics=metrics,
            scores=scores,
            summary=summary,
            report=report,
            configuration=config_dict,
            dataset_reference=artifact.dataset_reference,
            model_reference=artifact.model_reference,
            prompt_reference=artifact.prompt_reference,
            export_files=file_descriptors,
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"Successfully executed BenchmarkPipeline in {total_latency_ms:.2f}ms. Overall score: {summary.overall_score:.2f}/10.0"
        )
        return final_artifact
