"""Fine-Tuning Lineage Tracker for full reproducible provenance tracking."""

import hashlib
import json
from datetime import datetime

from tradesense_ml import __version__
from tradesense_ml.domain.schemas.distillation import DistillationArtifact
from tradesense_ml.domain.schemas.finetuning import (
    ModelLineage,
    TrainingConfiguration,
    TrainingExecution,
)


class FineTuningLineageTracker:
    """Lineage tracker assembling reproducible provenance metadata for fine-tuned models."""

    def compute_configuration_hash(self, config: TrainingConfiguration) -> str:
        """Compute deterministic SHA-256 hash of training configuration."""
        raw_dict = config.model_dump(mode="json")
        encoded = json.dumps(raw_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def generate_lineage(
        self,
        model_id: str,
        distillation_artifact: DistillationArtifact,
        config: TrainingConfiguration,
        backend_name: str,
        strategy_name: str,
        execution_context: TrainingExecution,
    ) -> ModelLineage:
        """Generate canonical ModelLineage container."""
        config_hash = self.compute_configuration_hash(config)
        dataset_ids = (
            [distillation_artifact.metadata.dataset_artifact_id]
            if distillation_artifact.metadata
            else []
        )
        teacher_model = (
            distillation_artifact.lineage.teacher_model
            if distillation_artifact.lineage
            else "teacher_llm_v1"
        )

        start_ts = execution_context.start_timestamp
        end_ts = execution_context.end_timestamp or datetime.utcnow()

        return ModelLineage(
            model_id=model_id,
            distillation_artifact_id=distillation_artifact.artifact_id,
            dataset_ids=dataset_ids,
            teacher_model=teacher_model,
            student_base_model=config.model_config_params.base_model_name_or_path,
            training_strategy=strategy_name,
            training_backend=backend_name,
            training_framework_version=execution_context.pytorch_version,
            configuration_hash=config_hash,
            random_seed=config.model_config_params.random_seed,
            repository_version=__version__,
            git_commit=execution_context.git_commit,
            execution_start_timestamp=start_ts,
            execution_end_timestamp=end_ts,
            hardware_summary={
                "gpu_count": execution_context.gpu_count,
                "gpu_models": execution_context.gpu_models,
                "os": execution_context.os_info,
            },
        )
