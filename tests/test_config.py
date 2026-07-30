"""Unit tests for configuration system."""

from pathlib import Path

from tradesense_ml.config.settings import AppSettings, load_hydra_config


def test_app_settings_defaults() -> None:
    """Test AppSettings default instantiations."""
    settings = AppSettings()
    assert settings.app_name == "TradeSense ML"
    assert settings.logging.level == "INFO"
    assert settings.storage.backend == "local"


def test_load_hydra_config() -> None:
    """Test loading Hydra configs from repository configs/ folder."""
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    settings = load_hydra_config(config_name="config", config_dir=config_dir)
    assert settings.app_name == "TradeSense ML"
    assert settings.storage.data_dir == "datasets"
