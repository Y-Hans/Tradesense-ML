"""Configuration models and Hydra initialization helpers."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field


class LoggingSettings(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    file_path: str = "outputs/logs/tradesense.log"


class StorageSettings(BaseModel):
    """Storage backend settings."""

    backend: str = "local"
    data_dir: str = "datasets"
    artifact_dir: str = "artifacts"


class ExperimentSettings(BaseModel):
    """Experiment tracking settings."""

    experiment_name: str = "tradesense_default"
    tracking_uri: str = "outputs/mlruns"


class ReviewSettings(BaseModel):
    """Review pipeline configuration settings."""

    strategy: str = "single"
    reviewer: str = "rule_based"
    approval_threshold: float = 7.0
    revision_threshold: float = 4.0
    criteria_version: str = "v1.0.0"
    review_version: str = "v1.0.0"
    review_timeout: float = 30.0
    quality_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "reasoning_quality": 0.25,
            "coaching_usefulness": 0.25,
            "educational_value": 0.20,
            "consistency": 0.15,
            "completeness": 0.15,
        }
    )


class DatasetSettings(BaseModel):
    """Dataset builder configuration settings."""

    dataset_id: str = "tradesense_sft_v1"
    dataset_version: str = "v1.0.0"
    dataset_format: str = "sft_instruction"
    export_formats: list[str] = Field(default_factory=lambda: ["jsonl", "json", "parquet"])
    split_ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    filtering: dict[str, Any] = Field(
        default_factory=lambda: {
            "only_approved": True,
            "min_quality_score": 7.0,
            "min_confidence": 0.0,
            "remove_duplicates": True,
            "remove_incomplete": True,
        }
    )
    seed: int = 42
    output_dir: str = "datasets"


class DistillationSettings(BaseModel):
    """Distillation pipeline configuration settings."""

    distillation_id: str = "tradesense_distillation_v1"
    version: str = "v1.0.0"
    distillation_strategy: str = "SFTStrategy"
    selection_strategy: str = "ThresholdSelection"
    selection_threshold: float = 7.0
    sampling_strategy: str = "UniformSampling"
    sampling_rate: float = 1.0
    curriculum_strategy: str = "StandardCurriculumStrategy"
    export_formats: list[str] = Field(default_factory=lambda: ["json", "jsonl", "parquet", "md"])
    validation_enabled: bool = True
    random_seed: int = 42
    output_dir: str = "outputs/distillation"


class AppSettings(BaseModel):
    """Global application settings."""

    app_name: str = "TradeSense ML"
    environment: str = "development"
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    distillation: DistillationSettings = Field(default_factory=DistillationSettings)


def load_hydra_config(
    config_name: str = "config", config_dir: str | Path | None = None
) -> AppSettings:
    """Load configuration files and convert DictConfig to AppSettings Pydantic model."""
    from hydra import compose, initialize_config_dir

    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[3] / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg: DictConfig = compose(config_name=config_name)
        raw_dict = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(raw_dict, dict)
        return AppSettings.model_validate(raw_dict)
