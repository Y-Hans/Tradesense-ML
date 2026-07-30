"""Unit tests for Typer CLI startup and command invocation."""

from typer.testing import CliRunner

from tradesense_ml.cli.main import app

runner = CliRunner()


def test_cli_version() -> None:
    """Test tsml --version flag."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "TradeSense ML CLI version" in result.stdout


def test_cli_help() -> None:
    """Test tsml --help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "dataset" in result.stdout
    assert "teacher" in result.stdout
    assert "review" in result.stdout
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "benchmark" in result.stdout
    assert "registry" in result.stdout
    assert "export" in result.stdout
    assert "serve" in result.stdout


def test_cli_dataset_list() -> None:
    """Test tsml dataset list command."""
    result = runner.invoke(app, ["dataset", "list"])
    assert result.exit_code == 0
    assert "Available Datasets in Registry" in result.stdout


def test_cli_benchmark_list() -> None:
    """Test tsml benchmark list command."""
    result = runner.invoke(app, ["benchmark", "list"])
    assert result.exit_code == 0
    assert "coaching_benchmark_v1" in result.stdout
