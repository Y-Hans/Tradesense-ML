"""Canonical domain models for dataset records, statistics, lineage, and manifests."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.lineage import DatasetSplit


class DatasetExample(BaseModel):
    """Canonical dataset record representation independent of export format."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., description="Unique dataset record identifier")
    instruction: str = Field(..., description="Task instruction or system prompt")
    input: str = Field(..., description="User input context (trade + market info)")
    output: str = Field(..., description="Expected target response text")
    messages: list[dict[str, str]] = Field(
        default_factory=list, description="Structured chat messages (system, user, assistant)"
    )
    prompt: str = Field(..., description="Combined prompt text for completion models")
    reasoning: str | None = Field(default=None, description="Optional reasoning chain text")
    format_type: str = Field(
        default="sft_instruction",
        description="Dataset format (sft_instruction, sft_chat, evaluation)",
    )
    review_info: dict[str, Any] = Field(
        default_factory=dict, description="Review metrics (score, verdict, reviewer)"
    )
    lineage: dict[str, Any] = Field(
        default_factory=dict, description="Source provenance lineage metadata"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class DatasetStatistics(BaseModel):
    """Summary statistics container for a dataset version or split."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(..., description="Dataset identifier")
    total_examples: int = Field(default=0, ge=0, description="Total examples evaluated")
    approved_examples: int = Field(default=0, ge=0, description="Number of approved examples")
    rejected_examples: int = Field(default=0, ge=0, description="Number of rejected examples")
    quality_score_mean: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Mean quality score"
    )
    quality_score_min: float = Field(default=0.0, ge=0.0, le=10.0, description="Min quality score")
    quality_score_max: float = Field(default=0.0, ge=0.0, le=10.0, description="Max quality score")
    quality_score_std: float = Field(
        default=0.0, ge=0.0, description="Quality score standard deviation"
    )
    average_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Average reviewer confidence"
    )
    average_response_length: float = Field(
        default=0.0, ge=0.0, description="Avg response length in characters"
    )
    average_prompt_length: float = Field(
        default=0.0, ge=0.0, description="Avg prompt length in characters"
    )
    dataset_size_bytes: int = Field(
        default=0, ge=0, description="Estimated total dataset size in bytes"
    )
    split_sizes: dict[str, int] = Field(
        default_factory=dict, description="Count of examples per split (train, validation, test)"
    )
    teacher_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution of source teacher models"
    )
    reviewer_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution of reviewer types/names"
    )
    version_info: dict[str, str] = Field(default_factory=dict, description="Component versions map")


class DatasetLineage(BaseModel):
    """Comprehensive dataset provenance and lineage tracking model."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(..., description="Dataset identifier")
    dataset_version: str = Field(..., description="Dataset semantic version string")
    generation_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Generation UTC timestamp"
    )
    random_seed: int = Field(default=42, description="Random seed used for deterministic builds")
    synthetic_generator_version: str = Field(
        default="v1.0.0", description="Synthetic generator version"
    )
    teacher_inference_version: str = Field(
        default="v1.0.0", description="Teacher inference pipeline version"
    )
    review_version: str = Field(default="v1.0.0", description="Review pipeline version")
    configuration_hash: str = Field(..., description="SHA-256 hash of build configuration")
    prompt_version: str = Field(default="v1", description="Prompt template version")
    review_criteria_version: str = Field(default="v1.0.0", description="Review criteria version")
    source_example_ids: list[str] = Field(
        default_factory=list, description="IDs of source reviewed examples"
    )


class DatasetManifest(BaseModel):
    """Immutable manifest describing an exported dataset release."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(..., description="Unique dataset ID")
    version: str = Field(..., description="Semantic version string")
    creation_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    dataset_format: str = Field(
        ..., description="Canonical format (sft_instruction, sft_chat, evaluation)"
    )
    split_sizes: dict[str, int] = Field(..., description="Sizes of individual splits")
    statistics_summary: dict[str, Any] = Field(..., description="Summary statistics dictionary")
    configuration_version: str = Field(default="v1.0.0", description="Config version string")
    lineage: dict[str, Any] = Field(..., description="Provenance lineage information")
    export_files: list[dict[str, Any]] = Field(
        default_factory=list, description="List of exported file manifests with checksums"
    )
    checksum: str = Field(..., description="SHA-256 checksum of manifest definition")


class DatasetVersion(BaseModel):
    """Dataset version definition."""

    model_config = ConfigDict(frozen=True)

    version_str: str = Field(..., description="Semantic version string, e.g. v1.0.0")
    release_notes: str = Field(default="", description="Release notes or description")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class DatasetMetadata(BaseModel):
    """Overall dataset metadata container."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Dataset name")
    description: str = Field(..., description="Dataset description")
    version: str = Field(..., description="Dataset version")
    license: str = Field(default="MIT", description="Dataset license")
    authors: list[str] = Field(default_factory=list, description="Authors or creators")
    split: DatasetSplit = Field(default=DatasetSplit.TRAIN, description="Primary split")
    extra: dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra metadata")


class DatasetArtifact(BaseModel):
    """Immutable dataset artifact representing a complete, versioned dataset release."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(..., description="Unique artifact identifier")
    dataset_metadata: DatasetMetadata = Field(..., description="Dataset metadata summary")
    lineage: DatasetLineage = Field(..., description="Provenance lineage information")
    statistics: DatasetStatistics = Field(..., description="Aggregate summary statistics")
    manifest: DatasetManifest = Field(..., description="Release manifest detailing export files")
    splits: dict[str, list[DatasetExample]] = Field(
        ..., description="Map of split name (train, validation, test) to DatasetExample lists"
    )
    export_files: list[dict[str, Any]] = Field(
        default_factory=list, description="Export file descriptors and checksums"
    )
