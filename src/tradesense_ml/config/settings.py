"""Configuration models and Hydra initialization helpers."""

from pathlib import Path

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


class AppSettings(BaseModel):
    """Global application settings."""

    app_name: str = "TradeSense ML"
    environment: str = "development"
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)


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
