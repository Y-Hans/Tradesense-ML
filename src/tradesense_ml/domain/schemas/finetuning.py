"""Canonical domain models for fine-tuning pipeline, model artifacts, training sessions, strategies, backends, checkpoints, evaluation, statistics, lineage, packaging, and reports."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelConfiguration(BaseModel):
    """Declarative fine-tuning and model configuration hyperparameters."""

    model_config = ConfigDict(frozen=True)

    base_model_name_or_path: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct", description="Base model identifier or local path"
    )
    architecture: str = Field(default="Qwen2ForCausalLM", description="Model architecture type")
    precision: str = Field(default="bfloat16", description="Compute precision (fp16, bf16, fp32)")
    use_lora: bool = Field(default=True, description="Whether Low-Rank Adaptation (LoRA) is used")
    lora_r: int = Field(default=16, ge=1, description="LoRA rank dimension")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA scaling factor alpha")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout rate")
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Target attention/MLP layers for LoRA adaptation",
    )
    learning_rate: float = Field(default=0.0002, gt=0.0, description="Initial learning rate")
    num_epochs: int = Field(default=3, ge=1, description="Total training epochs")
    per_device_train_batch_size: int = Field(
        default=4, ge=1, description="Training batch size per device"
    )
    per_device_eval_batch_size: int = Field(
        default=4, ge=1, description="Evaluation batch size per device"
    )
    gradient_accumulation_steps: int = Field(
        default=4, ge=1, description="Number of gradient accumulation steps"
    )
    warmup_ratio: float = Field(
        default=0.03, ge=0.0, le=1.0, description="Warmup ratio over total steps"
    )
    optimizer: str = Field(default="adamw_torch", description="Optimizer algorithm identifier")
    lr_scheduler: str = Field(default="cosine", description="Learning rate scheduler strategy")
    weight_decay: float = Field(default=0.01, ge=0.0, description="L2 weight decay factor")
    max_seq_length: int = Field(default=2048, ge=1, description="Maximum sequence length in tokens")
    checkpoint_interval_steps: int = Field(
        default=100, ge=1, description="Save checkpoint every N steps"
    )
    eval_interval_steps: int = Field(default=100, ge=1, description="Evaluate model every N steps")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    resume_from_checkpoint: str | None = Field(
        default=None, description="Path or checkpoint ID to resume from"
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional custom configuration parameters"
    )


class TrainingBackendConfiguration(BaseModel):
    """Configuration options specific to a execution training backend."""

    model_config = ConfigDict(frozen=True)

    backend_name: str = Field(
        ..., description="Backend identifier (mock, unsloth, axolotl, hf, trl)"
    )
    device: str = Field(default="cuda", description="Target execution device (cuda, cpu, mps)")
    distributed_mode: str = Field(
        default="none", description="Distributed training mode (none, ddp, fsdp, deepspeed)"
    )
    deepspeed_config: str | None = Field(
        default=None, description="Path to DeepSpeed configuration JSON if applicable"
    )
    backend_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Framework-specific backend keyword arguments"
    )


class TrainingConfiguration(BaseModel):
    """Complete top-level training workflow configuration."""

    model_config = ConfigDict(frozen=True)

    run_name: str = Field(..., description="Unique human-readable name for training run")
    strategy_name: str = Field(
        default="SFTTrainingStrategy", description="Training strategy identifier"
    )
    backend_config: TrainingBackendConfiguration = Field(
        ..., description="Training backend configuration"
    )
    model_config_params: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Model & training hyperparameters"
    )
    output_dir: str = Field(
        default="outputs/finetuning", description="Directory path for outputs and checkpoints"
    )
    export_formats: list[str] = Field(
        default_factory=lambda: ["directory", "json", "markdown", "manifest"],
        description="List of export format targets",
    )


class TrainingExecution(BaseModel):
    """Metadata describing the hardware and software execution environment."""

    model_config = ConfigDict(frozen=True)

    python_version: str = Field(..., description="Python runtime version")
    pytorch_version: str = Field(..., description="PyTorch library version")
    cuda_version: str | None = Field(default=None, description="CUDA driver/toolkit version")
    gpu_models: list[str] = Field(default_factory=list, description="GPUs attached to session")
    gpu_count: int = Field(default=0, ge=0, description="Total GPU devices used")
    os_info: str = Field(..., description="Operating System description")
    git_commit: str = Field(default="unknown", description="Repository git commit SHA")
    repository_version: str = Field(default="0.1.0", description="TradeSense ML package version")
    start_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Training start UTC timestamp"
    )
    end_timestamp: datetime | None = Field(
        default=None, description="Training completion UTC timestamp"
    )
    total_execution_seconds: float = Field(
        default=0.0, ge=0.0, description="Total wall-clock duration in seconds"
    )


class ModelCheckpoint(BaseModel):
    """Canonical model checkpoint record."""

    model_config = ConfigDict(frozen=True)

    checkpoint_id: str = Field(..., description="Unique checkpoint identifier (e.g. step-100)")
    step: int = Field(..., ge=0, description="Global step number")
    epoch: float = Field(..., ge=0.0, description="Epoch fraction completed")
    loss: float = Field(..., description="Training loss at checkpoint step")
    eval_loss: float | None = Field(default=None, description="Evaluation loss if computed")
    checkpoint_path: str = Field(..., description="Absolute or relative file path to checkpoint")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Checkpoint specific metric snapshot"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Checkpoint creation timestamp"
    )
    is_best: bool = Field(default=False, description="Whether this is marked as best checkpoint")


class CheckpointResult(BaseModel):
    """Metadata summary of checkpoint collection and selection."""

    model_config = ConfigDict(frozen=True)

    total_checkpoints_saved: int = Field(default=0, ge=0, description="Count of saved checkpoints")
    best_checkpoint: ModelCheckpoint | None = Field(
        default=None, description="Selected best checkpoint based on validation metric"
    )
    final_checkpoint: ModelCheckpoint | None = Field(
        default=None, description="Final checkpoint saved at training completion"
    )
    all_checkpoints: list[ModelCheckpoint] = Field(
        default_factory=list, description="Chronological list of all saved checkpoints"
    )
    resumed_from_path: str | None = Field(
        default=None, description="Checkpoint path resumed from if applicable"
    )


class TrainingMetrics(BaseModel):
    """Step and epoch level training metric histories."""

    model_config = ConfigDict(frozen=True)

    loss_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Step loss entries [{'step': int, 'loss': float, 'lr': float}]",
    )
    eval_loss_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Eval loss entries [{'step': int, 'eval_loss': float}]"
    )
    lr_history: list[float] = Field(
        default_factory=list, description="Learning rate trajectory over steps"
    )
    epoch_metrics: list[dict[str, Any]] = Field(
        default_factory=list, description="Aggregated metrics per completed epoch"
    )


class EvaluationResult(BaseModel):
    """Post-training evaluation engine outputs and validation metrics."""

    model_config = ConfigDict(frozen=True)

    eval_loss: float = Field(..., description="Final validation loss")
    perplexity: float = Field(..., description="Model perplexity calculated from eval loss")
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Task evaluation accuracy")
    token_accuracy: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Next token prediction accuracy"
    )
    convergence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Training stability & convergence index"
    )
    benchmark_eval_scores: dict[str, float] = Field(
        default_factory=dict, description="Benchmark evaluation task scores"
    )
    custom_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Custom evaluation hook outputs"
    )


class TrainingStatistics(BaseModel):
    """Aggregated training process statistics."""

    model_config = ConfigDict(frozen=True)

    total_steps: int = Field(default=0, ge=0, description="Total completed training steps")
    total_epochs: float = Field(default=0.0, ge=0.0, description="Total completed epochs")
    total_duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Total wall-clock training seconds"
    )
    total_parameters: int = Field(default=0, ge=0, description="Total model parameters count")
    trainable_parameters: int = Field(default=0, ge=0, description="Count of trainable parameters")
    trainable_percentage: float = Field(
        default=100.0, ge=0.0, le=100.0, description="Percentage of parameters trained"
    )
    initial_loss: float = Field(default=0.0, description="Starting training loss")
    final_loss: float = Field(default=0.0, description="Ending training loss")
    best_eval_loss: float | None = Field(default=None, description="Lowest evaluation loss")
    dataset_sample_count: int = Field(
        default=0, ge=0, description="Total training examples consumed"
    )
    tokens_processed: int = Field(default=0, ge=0, description="Total tokens processed in training")
    peak_gpu_memory_mb: float = Field(
        default=0.0, ge=0.0, description="Peak VRAM usage in megabytes"
    )


class TrainingBackendResult(BaseModel):
    """Raw result payload returned by a TrainingBackend."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="success", description="Backend execution status (success, failed)")
    model_weights_dir: str = Field(..., description="Path to directory containing saved weights")
    metrics: TrainingMetrics = Field(..., description="Captured metrics trajectory")
    checkpoints: list[ModelCheckpoint] = Field(
        default_factory=list, description="Checkpoints produced by backend"
    )
    framework_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Framework specific metadata output"
    )
    error_message: str | None = Field(default=None, description="Error details if status=failed")
    execution_time_seconds: float = Field(default=0.0, ge=0.0, description="Backend execution time")


class TrainingProcessingResult(BaseModel):
    """Canonical intermediate result holding training, checkpointing, evaluation, and backend outputs."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique training run ID")
    distillation_artifact_id: str = Field(..., description="Input DistillationArtifact ID")
    backend_result: TrainingBackendResult = Field(..., description="Output from TrainingBackend")
    checkpoint_result: CheckpointResult = Field(..., description="Collected checkpoint summary")
    evaluation_result: EvaluationResult = Field(..., description="Evaluation engine output")
    execution_context: TrainingExecution = Field(..., description="Hardware/software context")
    training_config: TrainingConfiguration = Field(
        ..., description="Resolved training configuration"
    )
    runtime_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Runtime session metadata"
    )
    warnings: list[str] = Field(default_factory=list, description="Warnings encountered during run")
    failure_info: dict[str, Any] | None = Field(
        default=None, description="Failure information if execution halted"
    )


class ModelMetadata(BaseModel):
    """Model identity and categorization metadata."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Unique canonical model artifact ID")
    model_name: str = Field(..., description="Human readable model name")
    base_model: str = Field(..., description="Base pretrained model reference")
    strategy_name: str = Field(..., description="Training strategy name used")
    backend_name: str = Field(..., description="Training backend framework name")
    version: str = Field(default="v1.0.0", description="Model version string")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Model creation timestamp"
    )
    tags: list[str] = Field(default_factory=list, description="Tags describing model capabilities")
    description: str = Field(default="", description="Detailed description of model fine-tuning")


class ModelSummary(BaseModel):
    """Concise summary of fine-tuning outcome."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Model artifact ID")
    base_model: str = Field(..., description="Base model name")
    strategy: str = Field(..., description="Strategy name")
    backend: str = Field(..., description="Backend name")
    final_loss: float = Field(..., description="Final training loss")
    eval_loss: float = Field(..., description="Final evaluation loss")
    total_epochs: float = Field(..., description="Completed epochs")
    total_steps: int = Field(..., description="Completed steps")
    best_checkpoint_id: str | None = Field(default=None, description="Best checkpoint ID")
    training_duration_seconds: float = Field(..., description="Training duration")


class ModelStatistics(BaseModel):
    """Aggregated statistics for final model artifact."""

    model_config = ConfigDict(frozen=True)

    training_stats: TrainingStatistics = Field(..., description="Process statistics")
    evaluation_result: EvaluationResult = Field(..., description="Evaluation statistics")
    memory_usage_mb: float = Field(default=0.0, ge=0.0, description="Peak VRAM footprint")
    parameter_count_summary: dict[str, int] = Field(
        default_factory=dict, description="Detailed parameter count breakdown"
    )


class ModelLineage(BaseModel):
    """Complete reproducible provenance tracking model for fine-tuned artifacts."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Model artifact ID")
    distillation_artifact_id: str = Field(..., description="Source DistillationArtifact ID")
    dataset_ids: list[str] = Field(default_factory=list, description="Source dataset IDs")
    teacher_model: str = Field(..., description="Teacher LLM used in distillation pipeline")
    student_base_model: str = Field(..., description="Base student LLM model fine-tuned")
    training_strategy: str = Field(..., description="Training strategy name")
    training_backend: str = Field(..., description="Backend engine name")
    training_framework_version: str = Field(
        default="1.0.0", description="Backend framework version"
    )
    configuration_hash: str = Field(..., description="SHA-256 hash of training configuration")
    random_seed: int = Field(..., description="Random seed used")
    repository_version: str = Field(..., description="TradeSense ML version")
    git_commit: str = Field(..., description="Git commit SHA")
    execution_start_timestamp: datetime = Field(..., description="Start timestamp")
    execution_end_timestamp: datetime = Field(..., description="End timestamp")
    hardware_summary: dict[str, Any] = Field(
        default_factory=dict, description="Summary of hardware utilized"
    )


class ModelManifest(BaseModel):
    """Immutable release manifest for ModelPackage files and checksums."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Model artifact ID")
    version: str = Field(default="v1.0.0", description="Model release version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    configuration_hash: str = Field(..., description="SHA-256 configuration hash")
    package_checksum: str = Field(..., description="SHA-256 checksum of entire package")
    files: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of file descriptors [{'path': str, 'size_bytes': int, 'sha256': str}]",
    )


class TrainingReport(BaseModel):
    """Structured fine-tuning execution report container."""

    model_config = ConfigDict(frozen=True)

    training_summary: dict[str, Any] = Field(..., description="High-level execution summary")
    loss_curves_summary: dict[str, Any] = Field(..., description="Loss curve analysis")
    checkpoint_summary: dict[str, Any] = Field(..., description="Checkpoint summary")
    evaluation_summary: dict[str, Any] = Field(..., description="Evaluation summary")
    configuration_summary: dict[str, Any] = Field(..., description="Configuration parameters")
    backend_summary: dict[str, Any] = Field(..., description="Backend details and environment")
    warnings: list[str] = Field(default_factory=list, description="Warnings identified during run")
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for deployment or further tuning"
    )
    resource_utilization: dict[str, Any] = Field(
        ..., description="GPU and CPU resource utilization statistics"
    )


class TrainingRun(BaseModel):
    """Descriptor for a single training execution run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique run ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Run timestamp")
    status: str = Field(default="success", description="Status (success, failed)")
    configuration_hash: str = Field(..., description="SHA-256 hash of configuration")


class ModelPackage(BaseModel):
    """Canonical physical deliverables model holding file paths, manifests, and checksums."""

    model_config = ConfigDict(frozen=True)

    package_id: str = Field(..., description="Unique package identifier")
    model_id: str = Field(..., description="Associated ModelArtifact ID")
    weights_path: str = Field(..., description="Path to directory/file containing model weights")
    tokenizer_path: str | None = Field(default=None, description="Path to tokenizer files")
    adapter_path: str | None = Field(
        default=None, description="Path to LoRA adapter files if separate"
    )
    config_path: str = Field(..., description="Path to model config file")
    manifest_path: str = Field(..., description="Path to manifest file")
    report_path: str = Field(..., description="Path to generated markdown report file")
    metadata_path: str = Field(..., description="Path to model metadata JSON file")
    manifest: ModelManifest = Field(..., description="Embedded ModelManifest")
    package_checksum: str = Field(..., description="SHA-256 checksum of complete package")
    file_list: list[str] = Field(
        default_factory=list, description="List of all package file relative paths"
    )


class ModelArtifact(BaseModel):
    """Top-level canonical, immutable domain artifact for fine-tuned models."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(..., description="Unique model artifact ID")
    metadata: ModelMetadata = Field(..., description="Model metadata")
    lineage: ModelLineage = Field(..., description="Complete reproducible provenance tracking")
    configuration: TrainingConfiguration = Field(
        ..., description="Training execution configuration"
    )
    summary: ModelSummary = Field(..., description="Executive summary")
    statistics: ModelStatistics = Field(..., description="Aggregated statistics")
    manifest: ModelManifest = Field(..., description="Release manifest")
    package: ModelPackage = Field(..., description="Physical deliverable model package")
    report: TrainingReport = Field(..., description="Formatted training report")
    export_files: list[dict[str, Any]] = Field(
        default_factory=list, description="Export file descriptors and checksums"
    )
