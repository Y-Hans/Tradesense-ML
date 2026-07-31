"""Comprehensive validation engine enforcing dataset compatibility, schema compliance, and artifact integrity."""

from typing import Any

from pydantic import BaseModel, Field

from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact, BenchmarkProfile
from tradesense_ml.domain.schemas.dataset import DatasetArtifact


class BenchmarkValidationReport(BaseModel):
    """Report detailing benchmark validation checks."""

    is_valid: bool = Field(..., description="Overall validation status")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")


class BenchmarkValidator:
    """Validator ensuring benchmark integrity before export."""

    @staticmethod
    def validate_dataset_compatibility(dataset_artifact: DatasetArtifact) -> list[str]:
        """Validate input dataset artifact compatibility."""
        errors = []
        if not dataset_artifact.artifact_id:
            errors.append("DatasetArtifact missing artifact_id.")
        if not dataset_artifact.splits:
            errors.append("DatasetArtifact contains empty splits dictionary.")

        total_examples = sum(len(examples) for examples in dataset_artifact.splits.values())
        if total_examples == 0:
            errors.append("DatasetArtifact contains 0 examples across all splits.")

        return errors

    @staticmethod
    def validate_configuration(
        profile: BenchmarkProfile, config_dict: dict[str, Any] | None = None
    ) -> list[str]:
        """Validate benchmark profile and hydra execution config."""
        errors = []
        if not profile.profile_id:
            errors.append("BenchmarkProfile missing profile_id.")
        if not profile.suite_names:
            errors.append("BenchmarkProfile does not specify any target suite_names.")
        return errors

    @staticmethod
    def validate_duplicate_ids(artifact: BenchmarkArtifact) -> list[str]:
        """Ensure no duplicate case or metric IDs exist in BenchmarkArtifact."""
        errors = []
        case_ids = [res.case_id for res in artifact.results]
        if len(case_ids) != len(set(case_ids)):
            errors.append("Duplicate case IDs detected in benchmark results.")

        metric_ids = [m.metric_id for m in artifact.metrics]
        if len(metric_ids) != len(set(metric_ids)):
            errors.append("Duplicate metric IDs detected in benchmark metrics.")

        return errors

    @staticmethod
    def validate_score_integrity(artifact: BenchmarkArtifact) -> list[str]:
        """Validate math and score range bounds (0.0 to 10.0)."""
        errors = []
        scores = artifact.scores
        if not (0.0 <= scores.overall_score <= 10.0):
            errors.append(f"Overall score {scores.overall_score} out of bounds [0.0, 10.0].")

        for cat, score in scores.category_scores.items():
            if not (0.0 <= score <= 10.0):
                errors.append(f"Category score for '{cat}' ({score}) out of bounds [0.0, 10.0].")

        for res in artifact.results:
            if not (0.0 <= res.score <= 10.0):
                errors.append(
                    f"Case score for '{res.case_id}' ({res.score}) out of bounds [0.0, 10.0]."
                )

        return errors

    @staticmethod
    def validate_artifact_completeness(artifact: BenchmarkArtifact) -> list[str]:
        """Check completeness of canonical BenchmarkArtifact fields."""
        errors = []
        if not artifact.artifact_id:
            errors.append("BenchmarkArtifact missing artifact_id.")
        if not artifact.metadata.benchmark_id:
            errors.append("BenchmarkArtifact metadata missing benchmark_id.")
        if not artifact.lineage.configuration_hash:
            errors.append("BenchmarkArtifact lineage missing configuration_hash.")
        if not artifact.results:
            errors.append("BenchmarkArtifact contains empty results list.")
        if not artifact.summary:
            errors.append("BenchmarkArtifact missing summary.")
        if not artifact.report:
            errors.append("BenchmarkArtifact missing report.")
        return errors

    @classmethod
    def validate_benchmark(
        cls,
        dataset_artifact: DatasetArtifact,
        profile: BenchmarkProfile,
        artifact: BenchmarkArtifact | None = None,
    ) -> BenchmarkValidationReport:
        """Run all validation rules and return BenchmarkValidationReport.

        Args:
            dataset_artifact: Target input dataset.
            profile: Declarative profile executed.
            artifact: Generated BenchmarkArtifact (optional for pre-export check).

        Returns:
            BenchmarkValidationReport object.
        """
        errors: list[str] = []
        warnings: list[str] = []

        errors.extend(cls.validate_dataset_compatibility(dataset_artifact))
        errors.extend(cls.validate_configuration(profile))

        if artifact:
            errors.extend(cls.validate_duplicate_ids(artifact))
            errors.extend(cls.validate_score_integrity(artifact))
            errors.extend(cls.validate_artifact_completeness(artifact))

        is_valid = len(errors) == 0
        return BenchmarkValidationReport(is_valid=is_valid, errors=errors, warnings=warnings)
