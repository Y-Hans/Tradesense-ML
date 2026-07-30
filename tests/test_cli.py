"""Unit tests for Typer CLI startup and command invocation."""

from pathlib import Path

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


def test_cli_dataset_commands(tmp_path: Path) -> None:
    """Test tsml dataset generate, validate, and preview commands end-to-end."""
    out_file = tmp_path / "synthetic_test.jsonl"

    # 1. Generate
    gen_result = runner.invoke(
        app, ["dataset", "generate", "-c", "5", "-o", str(out_file), "-s", "42"]
    )
    assert gen_result.exit_code == 0
    assert "Successfully generated and exported dataset" in gen_result.stdout
    assert out_file.exists()

    # 2. Validate
    val_result = runner.invoke(app, ["dataset", "validate", str(out_file)])
    assert val_result.exit_code == 0
    assert "VALIDATION PASSED" in val_result.stdout

    # 3. Preview
    prev_result = runner.invoke(app, ["dataset", "preview", str(out_file), "-n", "2"])
    assert prev_result.exit_code == 0
    assert "Sample #1" in prev_result.stdout


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
