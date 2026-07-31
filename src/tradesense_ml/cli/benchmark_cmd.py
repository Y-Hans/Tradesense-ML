"""Benchmark CLI subcommands for tsml."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tradesense_ml.benchmark.exporters import BenchmarkExporterManager
from tradesense_ml.benchmark.pipeline import BenchmarkPipeline
from tradesense_ml.benchmark.profiles import ProfileRegistry
from tradesense_ml.benchmark.validation import BenchmarkValidator
from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.dataset import (
    DatasetArtifact,
    DatasetExample,
    DatasetLineage,
    DatasetManifest,
    DatasetMetadata,
    DatasetStatistics,
)

app = typer.Typer(
    name="benchmark", help="Run, evaluate, validate, and export versioned benchmark suites."
)
console = Console()


def _create_mock_dataset_artifact(dataset_id: str = "tradesense_sft_v1") -> DatasetArtifact:
    """Helper to create a deterministic DatasetArtifact if no file path provided."""
    examples = [
        DatasetExample(
            example_id=f"ex_{i:03d}",
            instruction="Provide coaching advice on this trade.",
            input=f"Trade #{i}: Long BTCUSDT @ 65000, Stop Loss @ 64000. Risk ratio 2:1.",
            output="Excellent risk-to-reward setup. Keep your stop-loss tight and adhere to discipline.",
            prompt="User prompt context",
            review_info={"quality_score": 8.5 + (i % 2) * 0.5, "status": "approved"},
            lineage={"generator": "synthetic_v1"},
        )
        for i in range(10)
    ]
    return DatasetArtifact(
        artifact_id=dataset_id,
        dataset_metadata=DatasetMetadata(
            name=dataset_id,
            description="Synthetic coaching dataset release for benchmark execution",
            version="v1.0.0",
        ),
        lineage=DatasetLineage(
            dataset_id=dataset_id,
            dataset_version="v1.0.0",
            configuration_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        statistics=DatasetStatistics(
            dataset_id=dataset_id,
            total_examples=10,
            approved_examples=10,
            split_sizes={"train": 8, "validation": 1, "test": 1},
        ),
        manifest=DatasetManifest(
            dataset_id=dataset_id,
            version="v1.0.0",
            dataset_format="sft_instruction",
            split_sizes={"train": 8, "validation": 1, "test": 1},
            statistics_summary={"total": 10},
            lineage={},
            checksum="manifest_checksum",
        ),
        splits={"train": examples[:8], "validation": [examples[8]], "test": [examples[9]]},
    )


@app.command("run")
def run_benchmark(
    dataset_path: str | None = typer.Option(
        None, "--dataset", "-d", help="Path to DatasetArtifact JSON file"
    ),
    profile: str = typer.Option(
        "teacher_evaluation",
        "--profile",
        "-p",
        help="Benchmark profile name (e.g. teacher_evaluation, dataset_quality, prompt_evaluation)",
    ),
    benchmark_id: str = typer.Option(
        "coaching_benchmark_v1", "--benchmark-id", "-b", help="Unique benchmark release ID"
    ),
    target_model: str = typer.Option(
        "teacher_llm_v1", "--model", "-m", help="Target model under test"
    ),
    output_dir: str = typer.Option(
        "outputs/benchmarks", "--output-dir", "-o", help="Output directory for benchmark artifacts"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for deterministic execution"),
) -> None:
    """Run reproducible benchmark suite pipeline against a DatasetArtifact."""
    console.print(f"[bold green]Executing Benchmark Pipeline Profile: {profile}[/bold green]")
    console.print(f" Target Model: [cyan]{target_model}[/cyan]")
    console.print(f" Benchmark ID: [cyan]{benchmark_id}[/cyan]")

    try:
        if dataset_path and Path(dataset_path).exists():
            data_text = Path(dataset_path).read_text(encoding="utf-8")
            dataset_artifact = DatasetArtifact.model_validate_json(data_text)
            console.print(f" Loaded DatasetArtifact: [bold]{dataset_artifact.artifact_id}[/bold]")
        else:
            if dataset_path:
                console.print(
                    f"[yellow]Warning: Dataset path '{dataset_path}' not found. Using synthetic fixture artifact.[/yellow]"
                )
            dataset_artifact = _create_mock_dataset_artifact()
            console.print(
                f" Created synthetic DatasetArtifact: [bold]{dataset_artifact.artifact_id}[/bold]"
            )

        pipeline = BenchmarkPipeline(profile_name=profile)
        artifact = pipeline.run(
            input_data=dataset_artifact,
            benchmark_id=benchmark_id,
            target_model=target_model,
            output_dir=output_dir,
            seed=seed,
        )

        console.print("\n[bold green]Benchmark Execution Completed Successfully![/bold green]")
        console.print(
            f" Overall Score: [bold yellow]{artifact.scores.overall_score:.2f} / 10.0[/bold yellow]"
        )
        console.print(
            f" Ranking Tier: [bold cyan]{artifact.scores.ranking_info.get('tier', 'N/A')}[/bold cyan]"
        )
        console.print(f" Output Artifact ID: [bold]{artifact.artifact_id}[/bold]")
        console.print(f" Export Files Count: [bold]{len(artifact.export_files)}[/bold]\n")

    except Exception as e:
        console.print(f"[bold red]Error running benchmark pipeline: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command("report")
def generate_report(
    artifact_path: str = typer.Argument(..., help="Path to BenchmarkArtifact JSON file"),
) -> None:
    """Display human-readable report summary from BenchmarkArtifact."""
    path = Path(artifact_path)
    if not path.exists():
        console.print(f"[red]Error: Artifact file '{artifact_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    try:
        artifact = BenchmarkArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        console.print("\n[bold cyan]=== TradeSense ML Benchmark Report ===[/bold cyan]")
        console.print(f" Benchmark ID: [bold]{artifact.artifact_id}[/bold]")
        console.print(
            f" Overall Score: [bold yellow]{artifact.scores.overall_score:.2f} / 10.0[/bold yellow]"
        )
        console.print(f" Pass Rate: [green]{artifact.summary.pass_rate * 100:.1f}%[/green]")
        console.print(
            f" Ranking Tier: [bold]{artifact.scores.ranking_info.get('tier', 'N/A')}[/bold]"
        )

        table = Table(title="Category Scores Breakdown")
        table.add_column("Category", style="cyan")
        table.add_column("Score / 10.0", style="yellow")
        table.add_column("Status", style="green")

        for cat, sc in artifact.scores.category_scores.items():
            status = "PASS" if sc >= 6.0 else "FAIL"
            table.add_row(cat, f"{sc:.2f}", status)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to load BenchmarkArtifact report: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("validate")
def validate_benchmark(
    artifact_path: str = typer.Argument(..., help="Path to BenchmarkArtifact JSON file"),
) -> None:
    """Validate BenchmarkArtifact against schema compliance, duplicate IDs, and score integrity."""
    path = Path(artifact_path)
    if not path.exists():
        console.print(f"[red]Error: Artifact file '{artifact_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    try:
        artifact = BenchmarkArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        ds = _create_mock_dataset_artifact(artifact.metadata.dataset_id)
        report = BenchmarkValidator.validate_benchmark(
            dataset_artifact=ds, profile=artifact.profile, artifact=artifact
        )

        if report.is_valid:
            console.print(
                f"[bold green]BenchmarkArtifact '{artifact.artifact_id}' is valid and fully schema compliant![/bold green]"
            )
        else:
            console.print("[bold red]BenchmarkArtifact validation failed:[/bold red]")
            for err in report.errors:
                console.print(f" - [red]{err}[/red]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("export")
def export_benchmark(
    artifact_path: str = typer.Argument(..., help="Path to BenchmarkArtifact JSON file"),
    output_dir: str = typer.Option(
        "outputs/benchmarks", "--output-dir", "-o", help="Target output directory"
    ),
    formats: str = typer.Option(
        "json,jsonl,parquet,md", "--formats", "-f", help="Comma-separated export formats"
    ),
) -> None:
    """Export BenchmarkArtifact into JSON, JSONL, Parquet, and Markdown report formats."""
    path = Path(artifact_path)
    if not path.exists():
        console.print(f"[red]Error: Artifact file '{artifact_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    try:
        artifact = BenchmarkArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        fmt_list = [f.strip() for f in formats.split(",") if f.strip()]
        descriptors = BenchmarkExporterManager.export_artifact(
            artifact, output_dir=output_dir, formats=fmt_list
        )

        console.print(
            f"[bold green]Successfully exported BenchmarkArtifact '{artifact.artifact_id}':[/bold green]"
        )
        for desc in descriptors:
            console.print(
                f" - [{desc['format'].upper()}] {desc['path']} ({desc['size_bytes']} bytes)"
            )

    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("list")
def list_benchmarks() -> None:
    """List available benchmark profiles and suites."""
    console.print("[bold cyan]Available Benchmark Assets & Suites:[/bold cyan]")
    console.print(
        " • [bold yellow]coaching_benchmark_v1[/bold yellow] (v1.0.0, Default coaching evaluation benchmark suite)"
    )
    console.print("\n[bold cyan]Available Benchmark Profiles:[/bold cyan]")
    for p_id in ProfileRegistry.list_profiles():
        prof = ProfileRegistry.get(p_id)
        console.print(f" • [bold green]{p_id}[/bold green]: {prof.name}")
        console.print(f"   [dim]{prof.description}[/dim]")
