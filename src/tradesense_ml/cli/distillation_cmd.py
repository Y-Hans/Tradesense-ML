"""Distillation CLI subcommands for tsml."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tradesense_ml.distillation.exporters import DistillationExporterManager
from tradesense_ml.distillation.pipeline import DistillationPipeline
from tradesense_ml.distillation.strategies import DistillationStrategyRegistry
from tradesense_ml.distillation.validation import DistillationValidator
from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.dataset import (
    DatasetArtifact,
    DatasetExample,
    DatasetLineage,
    DatasetManifest,
    DatasetMetadata,
    DatasetStatistics,
)
from tradesense_ml.domain.schemas.distillation import DistillationArtifact

app = typer.Typer(
    name="distillation",
    help="Prepare, filter, sample, validate, and export distillation artifacts.",
)
console = Console()


def _create_mock_dataset_artifact(dataset_id: str = "tradesense_sft_v1") -> DatasetArtifact:
    """Helper to create a deterministic DatasetArtifact if no file path provided."""
    examples = [
        DatasetExample(
            example_id=f"ex_{i:03d}",
            instruction="Provide structured coaching on this trade execution.",
            input=f"Trade #{i}: Long BTCUSDT @ 65000, Stop Loss @ 64000. Risk-reward 2.5:1.",
            output=(
                f"Trade Coaching Feedback #{i}:\n"
                "1. Risk Analysis: Stop loss and position size adhere strictly to risk limits.\n"
                "2. Discipline Analysis: Excellent emotional control and rule adherence.\n"
                "3. Actionability: Next time, consider scaling out at key resistance levels."
            ),
            prompt="System prompt: You are an expert trading coach.",
            reasoning="Reasoning step: Trader followed rules and maintained stop loss.",
            review_info={"quality_score": 8.5 + (i % 3) * 0.5, "verdict": "approved"},
            lineage={"generator": "synthetic_v1"},
            metadata={"difficulty": 0.2 + 0.1 * i, "teacher_id": "teacher_llm_v1"},
        )
        for i in range(10)
    ]
    return DatasetArtifact(
        artifact_id=dataset_id,
        dataset_metadata=DatasetMetadata(
            name=dataset_id,
            description="Synthetic coaching dataset release for distillation",
            version="v1.0.0",
        ),
        lineage=DatasetLineage(
            dataset_id=dataset_id,
            dataset_version="v1.0.0",
            configuration_hash="hash_123456789",
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
            checksum="chk_12345",
        ),
        splits={"train": examples[:8], "validation": [examples[8]], "test": [examples[9]]},
    )


@app.command("run")
def run_distillation(
    dataset_path: str | None = typer.Option(
        None, "--dataset", "-d", help="Path to DatasetArtifact JSON file"
    ),
    benchmark_path: str | None = typer.Option(
        None, "--benchmark", "-b", help="Path to BenchmarkArtifact JSON file"
    ),
    distillation_id: str = typer.Option(
        "tradesense_distillation_v1", "--distillation-id", "-i", help="Distillation release ID"
    ),
    strategy: str = typer.Option(
        "SFTStrategy",
        "--strategy",
        "-s",
        help="Distillation strategy (SFTStrategy, DPOStrategy, ORPOStrategy, CurriculumStrategy, HybridStrategy)",
    ),
    selection_threshold: float = typer.Option(
        7.0, "--threshold", "-t", help="Quality selection threshold"
    ),
    output_dir: str = typer.Option(
        "outputs/distillation", "--output-dir", "-o", help="Output directory"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Run reproducible Distillation Pipeline consuming DatasetArtifact & BenchmarkArtifact."""
    console.print(f"[bold green]Executing Distillation Pipeline Strategy: {strategy}[/bold green]")
    console.print(f" Distillation ID: [cyan]{distillation_id}[/cyan]")

    try:
        if dataset_path and Path(dataset_path).exists():
            dataset_artifact = DatasetArtifact.model_validate_json(
                Path(dataset_path).read_text(encoding="utf-8")
            )
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

        benchmark_artifact: BenchmarkArtifact | None = None
        if benchmark_path and Path(benchmark_path).exists():
            benchmark_artifact = BenchmarkArtifact.model_validate_json(
                Path(benchmark_path).read_text(encoding="utf-8")
            )
            console.print(
                f" Loaded BenchmarkArtifact: [bold]{benchmark_artifact.artifact_id}[/bold]"
            )

        pipeline = DistillationPipeline(default_strategy=strategy)
        artifact = pipeline.run(
            input_data=dataset_artifact,
            benchmark_artifact=benchmark_artifact,
            distillation_id=distillation_id,
            distillation_strategy=strategy,
            selection_threshold=selection_threshold,
            output_dir=output_dir,
            seed=seed,
        )

        console.print("\n[bold green]Distillation Execution Completed Successfully![/bold green]")
        console.print(f" Output Artifact ID: [bold]{artifact.artifact_id}[/bold]")
        console.print(
            f" Sampled SFT Examples: [bold yellow]{artifact.summary.total_sampled_examples}[/bold yellow]"
        )
        console.print(
            f" Preference Pairs: [bold cyan]{artifact.summary.total_preference_pairs}[/bold cyan]"
        )
        console.print(
            f" Curriculum Stages: [bold green]{artifact.summary.total_curriculum_stages}[/bold green]"
        )
        console.print(f" Export Files Count: [bold]{len(artifact.export_files)}[/bold]\n")

    except Exception as e:
        console.print(f"[bold red]Error running distillation pipeline: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command("report")
def generate_report(
    artifact_path: str = typer.Argument(..., help="Path to DistillationArtifact JSON file"),
) -> None:
    """Display human-readable report summary from DistillationArtifact."""
    path = Path(artifact_path)
    if not path.exists():
        console.print(f"[red]Error: Artifact file '{artifact_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    try:
        artifact = DistillationArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        console.print("\n[bold cyan]=== TradeSense ML Distillation Report ===[/bold cyan]")
        console.print(f" Artifact ID: [bold]{artifact.artifact_id}[/bold]")
        console.print(f" Strategy: [green]{artifact.lineage.distillation_strategy}[/green]")
        console.print(
            f" Mean Quality Score: [bold yellow]{artifact.summary.overall_quality_mean:.2f} / 10.0[/bold yellow]"
        )

        table = Table(title="Curriculum Stages Breakdown")
        table.add_column("Stage", style="cyan")
        table.add_column("Examples Count", style="yellow")

        for stage in artifact.dataset.curriculum_stages:
            table.add_row(stage.name, str(stage.example_count))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to load DistillationArtifact report: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("validate")
def validate_distillation(
    artifact_path: str = typer.Argument(..., help="Path to DistillationArtifact JSON file"),
) -> None:
    """Validate DistillationArtifact against schema compliance, curriculum integrity, and preference pairs."""
    path = Path(artifact_path)
    if not path.exists():
        console.print(f"[red]Error: Artifact file '{artifact_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    try:
        artifact = DistillationArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        ds = _create_mock_dataset_artifact(artifact.metadata.dataset_artifact_id)
        report = DistillationValidator.validate_distillation(dataset_artifact=ds, artifact=artifact)

        if report.is_valid:
            console.print(
                f"[bold green]DistillationArtifact '{artifact.artifact_id}' is valid and fully schema compliant![/bold green]"
            )
        else:
            console.print("[bold red]DistillationArtifact validation failed:[/bold red]")
            for err in report.errors:
                console.print(f" - [red]{err}[/red]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("export")
def export_distillation(
    artifact_path: str = typer.Argument(..., help="Path to DistillationArtifact JSON file"),
    output_dir: str = typer.Option(
        "outputs/distillation", "--output-dir", "-o", help="Target output directory"
    ),
    formats: str = typer.Option(
        "json,jsonl,parquet,md", "--formats", "-f", help="Comma-separated export formats"
    ),
) -> None:
    """Export DistillationArtifact into JSON, JSONL, Parquet, and Markdown report formats."""
    path = Path(artifact_path)
    if not path.exists():
        console.print(f"[red]Error: Artifact file '{artifact_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    try:
        artifact = DistillationArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        fmt_list = [f.strip() for f in formats.split(",") if f.strip()]
        descriptors = DistillationExporterManager.export_artifact(
            artifact, output_dir=output_dir, formats=fmt_list
        )

        console.print(
            f"[bold green]Successfully exported DistillationArtifact '{artifact.artifact_id}':[/bold green]"
        )
        for desc in descriptors:
            console.print(
                f" - [{desc['format'].upper()}] {desc['path']} ({desc['size_bytes']} bytes)"
            )

    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("list")
def list_strategies() -> None:
    """List available distillation execution strategies."""
    console.print("[bold cyan]Available Distillation Execution Strategies:[/bold cyan]")
    for st_id in DistillationStrategyRegistry.list_strategies():
        console.print(f" • [bold green]{st_id}[/bold green]")
