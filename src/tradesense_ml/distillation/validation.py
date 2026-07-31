"""Distillation Validator for checking schema compliance, compatibility, and artifact completeness."""

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.dataset import DatasetArtifact
from tradesense_ml.domain.schemas.distillation import DistillationArtifact
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class DistillationValidationReport(BaseModel):
    """Validation outcome report."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(..., description="Validation outcome flag")
    errors: list[str] = Field(default_factory=list, description="Validation error messages")
    warnings: list[str] = Field(default_factory=list, description="Validation non-fatal warnings")


class DistillationValidator:
    """Validator performing pre and post execution integrity checks."""

    @classmethod
    def validate_distillation(
        cls,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        artifact: DistillationArtifact | None = None,
    ) -> DistillationValidationReport:
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Dataset compatibility
        if dataset_artifact.statistics.total_examples == 0:
            errors.append("Input DatasetArtifact has zero total examples.")

        # 2. Benchmark compatibility check
        if benchmark_artifact is not None:
            if benchmark_artifact.metadata.dataset_id != dataset_artifact.artifact_id:
                warnings.append(
                    f"BenchmarkArtifact dataset_id '{benchmark_artifact.metadata.dataset_id}' "
                    f"does not match input DatasetArtifact ID '{dataset_artifact.artifact_id}'."
                )

        # 3. Post-execution DistillationArtifact integrity checks
        if artifact is not None:
            # Schema / Field validation
            if not artifact.artifact_id:
                errors.append("DistillationArtifact artifact_id cannot be empty.")

            if artifact.dataset.total_examples == 0 and len(artifact.dataset.preference_pairs) == 0:
                warnings.append(
                    "DistillationArtifact dataset contains no SFT examples and no preference pairs."
                )

            # Selection integrity
            if len(artifact.statistics.selection_counts) == 0:
                errors.append("DistillationArtifact statistics missing selection_counts.")

            # Curriculum integrity
            for stage in artifact.dataset.curriculum_stages:
                if stage.example_count != len(stage.examples):
                    errors.append(
                        f"Curriculum stage '{stage.name}' example_count ({stage.example_count}) "
                        f"mismatches examples list length ({len(stage.examples)})."
                    )

            # Preference integrity
            for pair in artifact.dataset.preference_pairs:
                if not pair.chosen_response.strip():
                    errors.append(f"PreferencePair '{pair.pair_id}' chosen_response is empty.")

            # Manifest integrity
            if not artifact.manifest.checksum:
                errors.append("DistillationManifest checksum is missing.")

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"DistillationValidator identified {len(errors)} validation errors.")

        return DistillationValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )
