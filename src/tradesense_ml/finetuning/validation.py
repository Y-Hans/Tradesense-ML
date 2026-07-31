"""Fine-Tuning Validator checking artifact compatibility, configs, checkpoints, and packaging."""

from pathlib import Path

from tradesense_ml.domain.schemas.distillation import DistillationArtifact
from tradesense_ml.domain.schemas.finetuning import (
    ModelArtifact,
    ModelPackage,
    TrainingConfiguration,
)


class FineTuningValidator:
    """Validation engine verifying input compatibility, configurations, checkpoints, and artifacts."""

    def validate_distillation_artifact(self, artifact: DistillationArtifact) -> list[str]:
        """Validate input DistillationArtifact compatibility for fine-tuning."""
        issues: list[str] = []
        if not artifact.artifact_id:
            issues.append("DistillationArtifact missing artifact_id")
        if not artifact.dataset or (
            not artifact.dataset.sft_examples and not artifact.dataset.preference_pairs
        ):
            issues.append(
                "DistillationArtifact contains no training dataset sft_examples or preference_pairs"
            )
        return issues

    def validate_training_configuration(self, config: TrainingConfiguration) -> list[str]:
        """Validate TrainingConfiguration parameters."""
        issues: list[str] = []
        if not config.run_name:
            issues.append("TrainingConfiguration run_name cannot be empty")
        params = config.model_config_params
        if params.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        if params.num_epochs < 1:
            issues.append("num_epochs must be at least 1")
        if params.per_device_train_batch_size < 1:
            issues.append("per_device_train_batch_size must be at least 1")
        return issues

    def validate_model_package(self, package: ModelPackage) -> list[str]:
        """Validate completeness and file existence for ModelPackage."""
        issues: list[str] = []
        if not package.package_id:
            issues.append("ModelPackage missing package_id")

        for key, path_str in [
            ("config", package.config_path),
            ("manifest", package.manifest_path),
            ("metadata", package.metadata_path),
            ("report", package.report_path),
        ]:
            if not path_str or not Path(path_str).exists():
                issues.append(f"ModelPackage {key} file does not exist at: {path_str}")
        return issues

    def validate_model_artifact(self, artifact: ModelArtifact) -> list[str]:
        """Validate top-level canonical ModelArtifact structure and schema integrity."""
        issues: list[str] = []
        if not artifact.artifact_id:
            issues.append("ModelArtifact missing artifact_id")
        if not artifact.metadata or not artifact.metadata.model_id:
            issues.append("ModelArtifact metadata missing or invalid")
        if not artifact.lineage or not artifact.lineage.configuration_hash:
            issues.append("ModelArtifact lineage missing configuration_hash")
        if not artifact.package or not artifact.package.package_checksum:
            issues.append("ModelArtifact package missing checksum")

        # Validate nested package
        pkg_issues = self.validate_model_package(artifact.package)
        issues.extend(pkg_issues)
        return issues
