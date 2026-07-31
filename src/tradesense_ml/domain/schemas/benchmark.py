"""Canonical domain models for benchmark suites, runs, cases, metrics, scores, lineage, reports, profiles, and artifacts."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkProfile(BaseModel):
    """Declarative benchmark profile specifying target suites, cases, metrics, weights, and policies."""

    model_config = ConfigDict(frozen=True)

    profile_id: str = Field(..., description="Unique profile identifier")
    name: str = Field(..., description="Profile display name")
    description: str = Field(default="", description="Profile description")
    suite_names: list[str] = Field(
        default_factory=list, description="Target benchmark suites to execute"
    )
    enabled_case_ids: list[str] = Field(
        default_factory=list, description="Enabled benchmark case IDs (empty for all in suite)"
    )
    enabled_metric_ids: list[str] = Field(
        default_factory=list, description="Enabled metric IDs (empty for all)"
    )
    category_weights: dict[str, float] = Field(
        default_factory=dict, description="Category weights for scoring aggregation"
    )
    case_weights: dict[str, float] = Field(
        default_factory=dict, description="Case-level weights for scoring"
    )
    execution_policy: dict[str, Any] = Field(
        default_factory=dict, description="Execution policy options (retries, concurrency, seed)"
    )


class BenchmarkMetadata(BaseModel):
    """Metadata container for a benchmark execution."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str = Field(..., description="Unique benchmark execution identifier")
    name: str = Field(..., description="Benchmark name")
    description: str = Field(default="", description="Benchmark description")
    version: str = Field(default="v1.0.0", description="Benchmark release version")
    suite_name: str = Field(..., description="Primary benchmark suite evaluated")
    profile_name: str = Field(..., description="Profile used for execution")
    target_model: str = Field(default="teacher_llm_v1", description="Model under test identifier")
    student_model: str | None = Field(default=None, description="Optional student model ID")
    prompt_version: str = Field(default="v1", description="Prompt template version")
    dataset_id: str = Field(..., description="Input DatasetArtifact ID")
    dataset_version: str = Field(..., description="Input DatasetArtifact version")
    author: str = Field(default="TradeSense ML Team", description="Author or team")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    extra: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class BenchmarkMetric(BaseModel):
    """Atomic metric value representation."""

    model_config = ConfigDict(frozen=True)

    metric_id: str = Field(..., description="Unique metric identifier")
    name: str = Field(..., description="Metric display name")
    metric_type: str = Field(
        ...,
        description="Metric category/type (accuracy, pass_rate, quality_score, consistency_score, confidence, latency, token_usage, cost, response_length, prompt_length)",
    )
    value: float = Field(..., description="Calculated metric value")
    unit: str = Field(default="", description="Measurement unit (ms, tokens, %, points, USD, etc.)")
    min_value: float = Field(default=0.0, description="Minimum theoretical bound")
    max_value: float = Field(default=10.0, description="Maximum theoretical bound")
    confidence_interval: tuple[float, float] | None = Field(
        default=None, description="Optional 95% confidence interval bounds (lower, upper)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metric-specific details")


class BenchmarkCase(BaseModel):
    """Canonical model for a single benchmark evaluation case evaluating exactly one concern."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(..., description="Unique benchmark case identifier")
    name: str = Field(..., description="Benchmark case name")
    concern: str = Field(..., description="Single targeted evaluation concern")
    description: str = Field(default="", description="Detailed description of what is evaluated")
    weight: float = Field(default=1.0, ge=0.0, description="Importance weight")
    metric_ids: list[str] = Field(
        default_factory=list, description="IDs of metrics computed by this case"
    )


class BenchmarkExecutionResult(BaseModel):
    """Raw, un-scored observation and measurement output from executing a benchmark case."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(..., description="Benchmark case identifier")
    suite_id: str = Field(..., description="Parent benchmark suite identifier")
    raw_metrics: dict[str, float] = Field(
        default_factory=dict, description="Raw computed metric key-value pairs"
    )
    raw_observations: list[dict[str, Any]] = Field(
        default_factory=list, description="Raw per-example observations and measurements"
    )
    total_items_evaluated: int = Field(default=0, ge=0, description="Count of evaluated items")
    failed_items_count: int = Field(default=0, ge=0, description="Count of failed/flawed items")
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="Case execution latency in milliseconds"
    )
    status: str = Field(
        default="completed", description="Execution status (completed, failed, skipped)"
    )
    error_message: str | None = Field(default=None, description="Error message if execution failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution metadata")


class BenchmarkResult(BaseModel):
    """Standardized, scored result for a benchmark case after scoring engine evaluation."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(..., description="Benchmark case identifier")
    case_name: str = Field(..., description="Benchmark case name")
    concern: str = Field(..., description="Evaluated concern area")
    passed: bool = Field(..., description="Pass/fail verdict against threshold")
    score: float = Field(..., ge=0.0, le=10.0, description="Normalized score (0.0 to 10.0)")
    weight: float = Field(default=1.0, ge=0.0, description="Case weight applied during scoring")
    metrics: list[BenchmarkMetric] = Field(
        default_factory=list, description="Structured metrics associated with case"
    )
    details: dict[str, Any] = Field(default_factory=dict, description="Evaluation summary details")
    failure_reasons: list[str] = Field(
        default_factory=list, description="Reasons for failure if any"
    )
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings")


class BenchmarkScore(BaseModel):
    """Aggregated benchmark scores, category scores, breakdowns, and ranking tiers."""

    model_config = ConfigDict(frozen=True)

    overall_score: float = Field(
        ..., ge=0.0, le=10.0, description="Overall weighted benchmark score"
    )
    weighted_score: float = Field(..., ge=0.0, le=10.0, description="Weighted benchmark score")
    category_scores: dict[str, float] = Field(
        default_factory=dict, description="Scores grouped by category concern"
    )
    score_breakdown: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Detailed score breakdown per case and category"
    )
    ranking_info: dict[str, Any] = Field(
        default_factory=dict, description="Ranking metadata (tier, rating grade, percentile)"
    )


class BenchmarkSummary(BaseModel):
    """Executive summary of benchmark suite execution."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str = Field(..., description="Benchmark identifier")
    total_cases: int = Field(default=0, ge=0, description="Total benchmark cases evaluated")
    passed_cases: int = Field(default=0, ge=0, description="Count of passed cases")
    failed_cases: int = Field(default=0, ge=0, description="Count of failed cases")
    pass_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Case pass rate percentage (0..1)"
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Overall benchmark score"
    )
    category_scores: dict[str, float] = Field(
        default_factory=dict, description="Category score map"
    )
    execution_time_seconds: float = Field(
        default=0.0, ge=0.0, description="Total run wall-clock time"
    )
    ranking_tier: str = Field(default="N/A", description="Performance tier grade")


class BenchmarkRun(BaseModel):
    """Descriptor for a benchmark execution run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique run identifier")
    suite_name: str = Field(..., description="Target benchmark suite")
    profile_name: str = Field(..., description="Executed profile name")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Run timestamp")
    status: str = Field(default="success", description="Run completion status")
    configuration_hash: str = Field(..., description="SHA-256 configuration hash")


class BenchmarkSuite(BaseModel):
    """Definition model for a pluggable benchmark suite."""

    model_config = ConfigDict(frozen=True)

    suite_id: str = Field(..., description="Unique suite identifier")
    name: str = Field(..., description="Suite display name")
    version: str = Field(default="v1.0.0", description="Suite semantic version")
    description: str = Field(default="", description="Suite description")
    cases: list[BenchmarkCase] = Field(default_factory=list, description="Cases comprising suite")


class BenchmarkLineage(BaseModel):
    """Complete provenance tracking model for benchmark evaluations."""

    model_config = ConfigDict(frozen=True)

    benchmark_version: str = Field(default="v1.0.0", description="Benchmark engine version")
    dataset_artifact_id: str = Field(..., description="Input DatasetArtifact ID")
    dataset_version: str = Field(..., description="Input DatasetArtifact version")
    teacher_model: str = Field(..., description="Teacher model under evaluation")
    student_model: str | None = Field(
        default=None, description="Optional student model under evaluation"
    )
    prompt_version: str = Field(default="v1", description="Prompt template version evaluated")
    configuration_hash: str = Field(
        ..., description="SHA-256 hash of entire benchmark configuration"
    )
    metric_versions: dict[str, str] = Field(
        default_factory=dict, description="Version mapping of used metrics"
    )
    benchmark_suite_version: str = Field(default="v1.0.0", description="Version of benchmark suite")
    execution_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Execution timestamp"
    )
    random_seed: int = Field(default=42, description="Random seed used")


class BenchmarkReport(BaseModel):
    """Formatted report container independent of output exporters."""

    model_config = ConfigDict(frozen=True)

    overall_score: float = Field(..., ge=0.0, le=10.0, description="Overall score")
    category_scores: dict[str, float] = Field(..., description="Scores per category")
    metric_breakdown: list[dict[str, Any]] = Field(..., description="Detailed metric table data")
    failures: list[dict[str, Any]] = Field(..., description="List of failed benchmark cases")
    warnings: list[str] = Field(default_factory=list, description="Warnings identified")
    recommendations: list[str] = Field(
        default_factory=list, description="Actionable recommendations for improvement"
    )
    ranking: dict[str, Any] = Field(..., description="Ranking tier and grade details")
    configuration_summary: dict[str, Any] = Field(..., description="Summary of run configuration")
    dataset_summary: dict[str, Any] = Field(..., description="Summary of input dataset")
    model_summary: dict[str, Any] = Field(..., description="Summary of target model")


class BenchmarkArtifact(BaseModel):
    """Canonical, immutable benchmark release artifact representing full evaluation results."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(..., description="Unique benchmark artifact ID")
    metadata: BenchmarkMetadata = Field(..., description="Benchmark execution metadata")
    lineage: BenchmarkLineage = Field(..., description="Provenance tracking information")
    profile: BenchmarkProfile = Field(..., description="Declarative benchmark profile executed")
    suite_info: dict[str, Any] = Field(..., description="Summary of evaluated benchmark suites")
    execution_results: list[BenchmarkExecutionResult] = Field(
        ..., description="Raw un-scored case execution observations"
    )
    results: list[BenchmarkResult] = Field(..., description="Scored benchmark case results")
    metrics: list[BenchmarkMetric] = Field(..., description="All computed metrics across suite")
    scores: BenchmarkScore = Field(..., description="Aggregated scores and category breakdowns")
    summary: BenchmarkSummary = Field(..., description="Executive summary")
    report: BenchmarkReport = Field(..., description="Formatted benchmark report")
    configuration: dict[str, Any] = Field(..., description="Full Hydra execution configuration")
    dataset_reference: dict[str, Any] = Field(
        ..., description="Reference metadata of input DatasetArtifact"
    )
    model_reference: dict[str, Any] = Field(
        ..., description="Reference metadata of evaluated model"
    )
    prompt_reference: dict[str, Any] = Field(
        ..., description="Reference metadata of evaluated prompt"
    )
    export_files: list[dict[str, Any]] = Field(
        default_factory=list, description="List of exported file descriptors and checksums"
    )
