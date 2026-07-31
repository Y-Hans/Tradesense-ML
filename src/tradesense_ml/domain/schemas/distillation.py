"""Canonical domain models for distillation artifacts, lineage, metadata, statistics, processing results, preference pairs, and curriculum stages."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.lineage import DatasetSplit


class DistillationExample(BaseModel):
    """Canonical record for a distillation dataset example."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., description="Unique example identifier")
    instruction: str = Field(..., description="Task instruction or system prompt")
    input: str = Field(..., description="User input context (trade execution & market context)")
    output: str = Field(..., description="Teacher target response text")
    prompt: str = Field(..., description="Full prompt text supplied to model")
    messages: list[dict[str, str]] = Field(
        default_factory=list, description="Structured chat messages"
    )
    reasoning: str | None = Field(default=None, description="Optional reasoning chain text")
    quality_score: float = Field(
        default=8.0, ge=0.0, le=10.0, description="Evaluated quality score"
    )
    difficulty: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Normalized difficulty score (0=easy, 1=expert)"
    )
    quality_tier: str = Field(
        default="medium", description="Quality tier (easy, medium, hard, expert)"
    )
    teacher_id: str = Field(default="teacher_llm_v1", description="Source teacher model ID")
    format_type: str = Field(default="sft_instruction", description="Dataset format string")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom example metadata")


class PreferencePair(BaseModel):
    """Canonical DPO/ORPO preference pair object."""

    model_config = ConfigDict(frozen=True)

    pair_id: str = Field(..., description="Unique preference pair identifier")
    example_id: str = Field(..., description="Source example identifier")
    instruction: str = Field(..., description="Task instruction")
    input: str = Field(..., description="User input context")
    prompt: str = Field(..., description="Full prompt text")
    chosen_response: str = Field(..., description="Preferred/chosen teacher response")
    rejected_response: str = Field(..., description="Dispreferred/rejected response")
    preference_rationale: str = Field(
        default="", description="Detailed rationale explaining choice"
    )
    chosen_score: float = Field(
        default=9.0, ge=0.0, le=10.0, description="Score of chosen response"
    )
    rejected_score: float = Field(
        default=5.0, ge=0.0, le=10.0, description="Score of rejected response"
    )
    teacher_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata of teacher models"
    )
    benchmark_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark case and score context"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom pair metadata")


class CurriculumStage(BaseModel):
    """Curriculum stage container grouping examples by difficulty."""

    model_config = ConfigDict(frozen=True)

    stage_id: str = Field(..., description="Unique curriculum stage identifier")
    name: str = Field(..., description="Stage name (e.g. easy, medium, hard, expert)")
    description: str = Field(default="", description="Stage description")
    stage_order: int = Field(..., ge=1, description="Sequential stage order (1-indexed)")
    min_difficulty: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum difficulty bound"
    )
    max_difficulty: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Maximum difficulty bound"
    )
    examples: list[DistillationExample] = Field(
        default_factory=list, description="Examples assigned to stage"
    )
    example_ids: list[str] = Field(default_factory=list, description="Example IDs in stage")
    example_count: int = Field(default=0, ge=0, description="Count of examples in stage")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Stage metadata")


class SelectionResult(BaseModel):
    """Selection engine execution output."""

    model_config = ConfigDict(frozen=True)

    selected_example_ids: list[str] = Field(..., description="IDs of selected examples")
    selected_examples: list[DistillationExample] = Field(..., description="Selected examples")
    rejected_example_ids: list[str] = Field(
        default_factory=list, description="IDs of rejected examples"
    )
    strategy_name: str = Field(..., description="Name of selection strategy applied")
    selection_counts: dict[str, int] = Field(..., description="Selection metrics count map")
    threshold_applied: float = Field(default=7.0, description="Applied quality threshold")
    scores_map: dict[str, float] = Field(default_factory=dict, description="Per-example score map")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Selection metadata")


class SamplingResult(BaseModel):
    """Sampling engine execution output."""

    model_config = ConfigDict(frozen=True)

    sampled_example_ids: list[str] = Field(..., description="IDs of sampled examples")
    sampled_examples: list[DistillationExample] = Field(..., description="Sampled examples")
    strategy_name: str = Field(..., description="Sampling strategy applied")
    sample_size: int = Field(..., ge=0, description="Count of sampled items")
    sampling_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Sampling ratio applied")
    distribution_stats: dict[str, Any] = Field(
        default_factory=dict, description="Sampling distribution summary"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Sampling metadata")


class DistillationProcessingResult(BaseModel):
    """Intermediate canonical result produced by DistillationStrategy execution."""

    model_config = ConfigDict(frozen=True)

    selected_examples: list[DistillationExample] = Field(
        default_factory=list, description="Selected examples after selection & filtering"
    )
    rejected_examples: list[DistillationExample] = Field(
        default_factory=list, description="Examples rejected during selection or filtering"
    )
    sampled_examples: list[DistillationExample] = Field(
        default_factory=list, description="Sampled examples for final SFT release"
    )
    curriculum_stages: list[CurriculumStage] = Field(
        default_factory=list, description="Generated curriculum stages"
    )
    preference_pairs: list[PreferencePair] = Field(
        default_factory=list, description="Generated preference pairs"
    )
    selection_result: SelectionResult = Field(..., description="Raw selection engine output")
    sampling_result: SamplingResult = Field(..., description="Raw sampling engine output")
    filtering_stats: dict[str, Any] = Field(
        default_factory=dict, description="Filtering stats breakdown"
    )
    processing_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary processing metadata"
    )


class DistillationDataset(BaseModel):
    """Container for distilled dataset outputs and splits."""

    model_config = ConfigDict(frozen=True)

    sft_examples: list[DistillationExample] = Field(
        default_factory=list, description="Supervised fine-tuning examples"
    )
    preference_pairs: list[PreferencePair] = Field(
        default_factory=list, description="Preference dataset pairs"
    )
    curriculum_stages: list[CurriculumStage] = Field(
        default_factory=list, description="Curriculum stages"
    )
    split_name: DatasetSplit = Field(
        default=DatasetSplit.TRAIN, description="Primary dataset split"
    )
    total_examples: int = Field(default=0, ge=0, description="Total SFT examples count")
    total_preference_pairs: int = Field(default=0, ge=0, description="Total preference pairs count")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")


class DistillationMetadata(BaseModel):
    """Metadata summary container for distillation artifact."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(..., description="Unique distillation artifact identifier")
    name: str = Field(..., description="Distillation release name")
    description: str = Field(default="", description="Release description")
    version: str = Field(default="v1.0.0", description="Distillation semantic version")
    dataset_artifact_id: str = Field(..., description="Source DatasetArtifact ID")
    benchmark_artifact_id: str = Field(..., description="Source BenchmarkArtifact ID")
    teacher_model: str = Field(default="teacher_llm_v1", description="Primary teacher model")
    prompt_version: str = Field(default="v1", description="Prompt template version")
    author: str = Field(default="TradeSense ML Team", description="Author or team")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    extra: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class DistillationLineage(BaseModel):
    """Comprehensive lineage and provenance tracking model for distillation runs."""

    model_config = ConfigDict(frozen=True)

    dataset_artifact_id: str = Field(..., description="Input DatasetArtifact ID")
    benchmark_artifact_id: str = Field(..., description="Input BenchmarkArtifact ID")
    teacher_model: str = Field(..., description="Teacher model identifier")
    prompt_version: str = Field(default="v1", description="Prompt template version")
    selection_strategy: str = Field(..., description="Selection strategy name used")
    sampling_strategy: str = Field(..., description="Sampling strategy name used")
    curriculum_strategy: str = Field(..., description="Curriculum strategy name used")
    distillation_strategy: str = Field(
        default="SFTStrategy", description="Top-level distillation strategy name"
    )
    configuration_hash: str = Field(..., description="SHA-256 hash of entire run configuration")
    random_seed: int = Field(default=42, description="Random seed used")
    execution_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Execution UTC timestamp"
    )
    repository_version: str = Field(default="v1.0.0", description="Repository version string")


class DistillationConfiguration(BaseModel):
    """Declarative execution configuration model for distillation runs."""

    model_config = ConfigDict(frozen=True)

    distillation_strategy: str = Field(default="SFTStrategy", description="Distillation strategy")
    selection_strategy: str = Field(default="ThresholdSelection", description="Selection strategy")
    selection_thresholds: dict[str, float] = Field(
        default_factory=dict, description="Quality thresholds"
    )
    sampling_strategy: str = Field(default="UniformSampling", description="Sampling strategy")
    curriculum_strategy: str = Field(
        default="StandardCurriculumStrategy", description="Curriculum strategy"
    )
    preference_config: dict[str, Any] = Field(
        default_factory=dict, description="Preference pair generation config"
    )
    export_formats: list[str] = Field(
        default_factory=lambda: ["json", "jsonl", "parquet", "md"],
        description="Target export formats",
    )
    validation_enabled: bool = Field(default=True, description="Enforce validation checks")
    random_seed: int = Field(default=42, description="Random seed")
    output_dir: str = Field(default="outputs/distillation", description="Target output directory")
    extra: dict[str, Any] = Field(default_factory=dict, description="Custom options")


class DistillationStatistics(BaseModel):
    """Aggregate statistics computed from DistillationProcessingResult."""

    model_config = ConfigDict(frozen=True)

    selection_counts: dict[str, int] = Field(
        default_factory=dict, description="Selection and approval counts"
    )
    rejection_counts: dict[str, int] = Field(
        default_factory=dict, description="Rejection reason breakdown"
    )
    sampling_statistics: dict[str, Any] = Field(
        default_factory=dict, description="Sampling size and ratio statistics"
    )
    curriculum_distribution: dict[str, int] = Field(
        default_factory=dict, description="Example counts per curriculum stage"
    )
    preference_counts: dict[str, int] = Field(
        default_factory=dict, description="Preference pair generation metrics"
    )
    teacher_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution across teacher models"
    )
    difficulty_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution across difficulty tiers"
    )
    quality_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution across quality score bins"
    )
    dataset_size_bytes: int = Field(default=0, ge=0, description="Estimated byte size of dataset")
    total_examples: int = Field(default=0, ge=0, description="Total SFT examples")
    token_estimates: dict[str, int] = Field(
        default_factory=dict, description="Estimated prompt and response token counts"
    )


class DistillationSummary(BaseModel):
    """Executive summary of distillation run."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(..., description="Artifact identifier")
    total_input_examples: int = Field(default=0, ge=0, description="Input dataset example count")
    total_selected_examples: int = Field(default=0, ge=0, description="Selected example count")
    total_sampled_examples: int = Field(default=0, ge=0, description="Sampled example count")
    total_preference_pairs: int = Field(
        default=0, ge=0, description="Generated preference pairs count"
    )
    total_curriculum_stages: int = Field(
        default=0, ge=0, description="Generated curriculum stages count"
    )
    overall_quality_mean: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Mean quality score of selected dataset"
    )
    execution_time_seconds: float = Field(
        default=0.0, ge=0.0, description="Total execution wall time"
    )


class DistillationManifest(BaseModel):
    """Immutable release manifest describing exported distillation files."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(..., description="Distillation artifact ID")
    version: str = Field(default="v1.0.0", description="Semantic version string")
    creation_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    statistics_summary: dict[str, Any] = Field(..., description="Summary statistics map")
    configuration_hash: str = Field(..., description="SHA-256 configuration hash")
    lineage: dict[str, Any] = Field(..., description="Lineage summary dictionary")
    export_files: list[dict[str, Any]] = Field(
        default_factory=list, description="Export file descriptors with checksums"
    )
    checksum: str = Field(..., description="Manifest SHA-256 checksum")


class DistillationRun(BaseModel):
    """Descriptor for a distillation execution run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique run identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Run timestamp")
    status: str = Field(default="success", description="Completion status (success, failed)")
    configuration_hash: str = Field(..., description="SHA-256 configuration hash")


class DistillationReport(BaseModel):
    """Structured distillation report container independent of exporters."""

    model_config = ConfigDict(frozen=True)

    selection_summary: dict[str, Any] = Field(..., description="Selection summary")
    filtering_summary: dict[str, Any] = Field(..., description="Filtering summary")
    sampling_summary: dict[str, Any] = Field(..., description="Sampling summary")
    curriculum_summary: dict[str, Any] = Field(..., description="Curriculum summary")
    preference_summary: dict[str, Any] = Field(..., description="Preference summary")
    statistics: DistillationStatistics = Field(..., description="Aggregated statistics")
    warnings: list[str] = Field(default_factory=list, description="Warnings identified")
    recommendations: list[str] = Field(
        default_factory=list, description="Actionable recommendations for fine-tuning"
    )
    configuration_summary: dict[str, Any] = Field(..., description="Run configuration summary")
    dataset_summary: dict[str, Any] = Field(..., description="Input dataset summary")
    benchmark_summary: dict[str, Any] = Field(..., description="Input benchmark summary")


class DistillationArtifact(BaseModel):
    """Canonical, immutable distillation release artifact representing prepared fine-tuning datasets."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(..., description="Unique distillation artifact ID")
    metadata: DistillationMetadata = Field(..., description="Distillation metadata summary")
    lineage: DistillationLineage = Field(..., description="Provenance tracking information")
    configuration: DistillationConfiguration = Field(
        ..., description="Declarative execution configuration"
    )
    summary: DistillationSummary = Field(..., description="Executive summary")
    statistics: DistillationStatistics = Field(..., description="Aggregated summary statistics")
    manifest: DistillationManifest = Field(..., description="Release manifest")
    dataset: DistillationDataset = Field(
        ..., description="Canonical distillation dataset container"
    )
    report: DistillationReport = Field(..., description="Formatted distillation report")
    export_files: list[dict[str, Any]] = Field(
        default_factory=list, description="Export file descriptors and checksums"
    )
    dataset_reference: dict[str, Any] = Field(
        default_factory=dict, description="Reference metadata of input DatasetArtifact"
    )
    benchmark_reference: dict[str, Any] = Field(
        default_factory=dict, description="Reference metadata of input BenchmarkArtifact"
    )
