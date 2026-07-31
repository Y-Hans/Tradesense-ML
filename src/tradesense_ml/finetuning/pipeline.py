"""FineTuningPipeline orchestrating model training, evaluation, lineage, reporting, packaging, and exports."""

import json
from pathlib import Path
from typing import Any

from tradesense_ml.domain.schemas.distillation import DistillationArtifact
from tradesense_ml.domain.schemas.finetuning import (
    ModelArtifact,
    ModelMetadata,
    TrainingBackendConfiguration,
    TrainingConfiguration,
    TrainingProcessingResult,
)
from tradesense_ml.finetuning.checkpoint import CheckpointManager
from tradesense_ml.finetuning.evaluation import FineTuningEvaluationEngine
from tradesense_ml.finetuning.exporters import ModelExporter
from tradesense_ml.finetuning.lineage import FineTuningLineageTracker
from tradesense_ml.finetuning.packaging import ModelPackager
from tradesense_ml.finetuning.reporting import FineTuningReporter
from tradesense_ml.finetuning.runner import FineTuningRunner
from tradesense_ml.finetuning.statistics import FineTuningStatisticsGenerator
from tradesense_ml.finetuning.validation import FineTuningValidator
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.pipelines.base import BasePipeline

logger = get_logger()


class FineTuningPipeline(BasePipeline[DistillationArtifact, ModelArtifact]):
    """Backend-agnostic Fine-Tuning Pipeline orchestrator consuming DistillationArtifact and producing canonical ModelArtifact.

    Architecture Flow:
    DistillationArtifact
            │
            ▼
    FineTuningPipeline
            │
            ▼
    FineTuningRunner
            │
            ▼
    TrainingSession
            │
            ▼
    TrainingStrategy
            │
            ▼
    TrainingBackend
            │
            ▼
    TrainingBackendResult
            │
            ▼
    TrainingProcessingResult
            │
            ├────────► Statistics
            ├────────► Evaluation
            ├────────► Lineage
            ├────────► Reporting
            ├────────► Validation
            └────────► Packaging
                        │
                        ▼
                   ModelPackage
                        │
                        ▼
                   ModelArtifact
    """

    def __init__(
        self,
        runner: FineTuningRunner | None = None,
        validator: FineTuningValidator | None = None,
        eval_engine: FineTuningEvaluationEngine | None = None,
        stats_generator: FineTuningStatisticsGenerator | None = None,
        lineage_tracker: FineTuningLineageTracker | None = None,
        reporter: FineTuningReporter | None = None,
        exporter: ModelExporter | None = None,
    ) -> None:
        super().__init__(pipeline_name="finetuning_pipeline")
        self.runner = runner or FineTuningRunner()
        self.validator = validator or FineTuningValidator()
        self.eval_engine = eval_engine or FineTuningEvaluationEngine()
        self.stats_generator = stats_generator or FineTuningStatisticsGenerator()
        self.lineage_tracker = lineage_tracker or FineTuningLineageTracker()
        self.reporter = reporter or FineTuningReporter()
        self.exporter = exporter or ModelExporter()

    def run(
        self,
        input_data: DistillationArtifact,
        config: TrainingConfiguration | None = None,
        output_dir: str = "outputs/finetuning",
        resume_from_checkpoint: str | None = None,
        **kwargs: Any,
    ) -> ModelArtifact:
        """Execute fine-tuning pipeline end-to-end.

        Args:
            input_data: Canonical DistillationArtifact input.
            config: Optional TrainingConfiguration (if omitted, constructed from kwargs or default).
            output_dir: Directory where run artifacts are stored.
            resume_from_checkpoint: Path to checkpoint to resume training from.
            **kwargs: Extra runtime parameter overrides.

        Returns:
            Canonical ModelArtifact container.
        """
        logger.info(
            f"Starting FineTuningPipeline run for DistillationArtifact: {input_data.artifact_id}"
        )

        # 1. Validate DistillationArtifact input compatibility
        compat_issues = self.validator.validate_distillation_artifact(input_data)
        if compat_issues:
            raise ValueError(f"DistillationArtifact validation failed: {'; '.join(compat_issues)}")

        # 2. Build or resolve TrainingConfiguration
        if config is None:
            backend_name = kwargs.get("backend_name", "mock")
            strategy_name = kwargs.get("strategy_name", "SFTTrainingStrategy")
            run_name = kwargs.get("run_name", f"run-{input_data.artifact_id}")

            backend_cfg = TrainingBackendConfiguration(backend_name=backend_name)
            config = TrainingConfiguration(
                run_name=run_name,
                strategy_name=strategy_name,
                backend_config=backend_cfg,
                output_dir=output_dir,
            )

        # Validate configuration
        cfg_issues = self.validator.validate_training_configuration(config)
        if cfg_issues:
            raise ValueError(f"TrainingConfiguration validation failed: {'; '.join(cfg_issues)}")

        run_id = f"ft-{config.run_name}"
        run_output_dir = str(Path(output_dir) / run_id)

        # 3. Execute training via FineTuningRunner & TrainingSession
        backend_result, session = self.runner.run(
            run_id=run_id,
            distillation_artifact=input_data,
            config=config,
            output_dir=run_output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        # 4. Checkpoint management
        ckpt_manager = CheckpointManager(output_dir=run_output_dir)
        checkpoint_result = ckpt_manager.compile_checkpoint_result(
            checkpoints=backend_result.checkpoints,
            resumed_from_path=resume_from_checkpoint,
        )

        # 5. Post-training evaluation engine execution
        evaluation_result = self.eval_engine.evaluate(backend_result)

        # 6. Canonical intermediate object: TrainingProcessingResult
        processing_result = TrainingProcessingResult(
            run_id=run_id,
            distillation_artifact_id=input_data.artifact_id,
            backend_result=backend_result,
            checkpoint_result=checkpoint_result,
            evaluation_result=evaluation_result,
            execution_context=session.execution_context,
            training_config=config,
            runtime_metadata=session.runtime_metadata,
            warnings=[],
            failure_info=None,
        )

        # 7. Generate statistics
        tr_stats = self.stats_generator.generate_training_statistics(
            distillation_artifact=input_data,
            backend_result=backend_result,
            model_config=config.model_config_params,
            checkpoint_result=checkpoint_result,
        )
        model_stats = self.stats_generator.generate_model_statistics(
            training_stats=tr_stats, eval_result=evaluation_result
        )

        model_id = f"model-{run_id}"

        # 8. Generate model summary & metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=config.run_name,
            base_model=config.model_config_params.base_model_name_or_path,
            strategy_name=config.strategy_name,
            backend_name=config.backend_config.backend_name,
            version="v1.0.0",
            tags=["fine-tuned", config.strategy_name.lower(), config.backend_config.backend_name],
            description=f"Fine-tuned model derived from DistillationArtifact {input_data.artifact_id}",
        )

        model_summary = self.stats_generator.generate_model_summary(
            model_id=model_id,
            base_model=config.model_config_params.base_model_name_or_path,
            strategy=config.strategy_name,
            backend=config.backend_config.backend_name,
            training_stats=tr_stats,
            eval_result=evaluation_result,
            checkpoint_result=checkpoint_result,
        )

        # 9. Lineage tracking
        lineage = self.lineage_tracker.generate_lineage(
            model_id=model_id,
            distillation_artifact=input_data,
            config=config,
            backend_name=config.backend_config.backend_name,
            strategy_name=config.strategy_name,
            execution_context=session.execution_context,
        )

        # 10. Generate structured report
        report = self.reporter.generate_report(processing_result=processing_result)

        # 11. Packaging -> ModelPackage
        packager = ModelPackager(output_dir=run_output_dir)
        model_package = packager.package_model(
            model_id=model_id,
            processing_result=processing_result,
            metadata=metadata,
            lineage=lineage,
            statistics=model_stats,
            report=report,
        )

        # 12. Build top-level ModelArtifact
        artifact = ModelArtifact(
            artifact_id=model_id,
            metadata=metadata,
            lineage=lineage,
            configuration=config,
            summary=model_summary,
            statistics=model_stats,
            manifest=model_package.manifest,
            package=model_package,
            report=report,
            export_files=[],
        )

        # 13. Export outputs
        export_descriptors = self.exporter.export(
            artifact=artifact,
            output_dir=run_output_dir,
            export_formats=config.export_formats,
        )

        # Re-instantiate artifact with export file descriptors
        final_artifact = ModelArtifact(
            artifact_id=artifact.artifact_id,
            metadata=artifact.metadata,
            lineage=artifact.lineage,
            configuration=artifact.configuration,
            summary=artifact.summary,
            statistics=artifact.statistics,
            manifest=artifact.manifest,
            package=artifact.package,
            report=artifact.report,
            export_files=export_descriptors,
        )

        # 14. Final validation of ModelArtifact
        art_issues = self.validator.validate_model_artifact(final_artifact)
        if art_issues:
            logger.warning(f"ModelArtifact validation warnings: {'; '.join(art_issues)}")

        logger.info(f"Successfully generated ModelArtifact: {final_artifact.artifact_id}")
        return final_artifact

    @classmethod
    def load_distillation_artifact(cls, file_path: str) -> DistillationArtifact:
        """Helper method to load DistillationArtifact from JSON file."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return DistillationArtifact.model_validate(data)
