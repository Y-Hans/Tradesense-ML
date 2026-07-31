"""Fine-Tuning CLI subcommands for tsml."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tradesense_ml.domain.schemas.distillation import (
    CurriculumStage,
    DistillationArtifact,
    DistillationConfiguration,
    DistillationDataset,
    DistillationExample,
    DistillationLineage,
    DistillationManifest,
    DistillationMetadata,
    DistillationReport,
    DistillationStatistics,
    DistillationSummary,
)
from tradesense_ml.domain.schemas.finetuning import (
    ModelArtifact,
    TrainingBackendConfiguration,
    TrainingConfiguration,
)
from tradesense_ml.finetuning.checkpoint import CheckpointManager
from tradesense_ml.finetuning.exporters import ModelExporter
from tradesense_ml.finetuning.pipeline import FineTuningPipeline
from tradesense_ml.finetuning.reporting import FineTuningReporter
from tradesense_ml.finetuning.validation import FineTuningValidator

app = typer.Typer(
    name="finetune",
    help="Orchestrate fine-tuning pipeline execution, validation, reporting, packaging, and checkpointing.",
)
console = Console()


def _create_mock_distillation_artifact(artifact_id: str = "dist_v1") -> DistillationArtifact:
    """Helper creating a deterministic mock DistillationArtifact for testing and dry-run CLI execution."""
    examples = [
        DistillationExample(
            example_id=f"ex_{i:03d}",
            instruction="Provide structured coaching on this trade execution.",
            input=f"Trade #{i}: Long BTCUSDT @ 65000, Stop Loss @ 64000.",
            output=f"Trade Coaching Feedback #{i}: Adhered to risk management rules.",
            prompt="You are a trading coach.",
            quality_score=8.5 + (i % 3) * 0.5,
            difficulty=min(1.0, 0.1 + 0.05 * i),
        )
        for i in range(10)
    ]
    dist_dataset = DistillationDataset(
        sft_examples=examples,
        total_examples=len(examples),
        curriculum_stages=[
            CurriculumStage(
                stage_id="stage_1",
                name="easy",
                stage_order=1,
                examples=examples[:5],
                example_ids=[e.example_id for e in examples[:5]],
                example_count=5,
            ),
            CurriculumStage(
                stage_id="stage_2",
                name="medium",
                stage_order=2,
                examples=examples[5:],
                example_ids=[e.example_id for e in examples[5:]],
                example_count=5,
            ),
        ],
    )
    meta = DistillationMetadata(
        artifact_id=artifact_id,
        name=artifact_id,
        dataset_artifact_id="dataset_v1",
        benchmark_artifact_id="bench_v1",
        description="Mock distillation release",
    )
    lineage = DistillationLineage(
        dataset_artifact_id="dataset_v1",
        benchmark_artifact_id="bench_v1",
        teacher_model="teacher_llm_v1",
        selection_strategy="ThresholdSelection",
        sampling_strategy="UniformSampling",
        curriculum_strategy="StandardCurriculumStrategy",
        configuration_hash="cfg_hash_mock_123",
    )
    cfg = DistillationConfiguration(distillation_strategy="SFTStrategy")
    summary = DistillationSummary(
        artifact_id=artifact_id,
        total_input_examples=len(examples),
        total_selected_examples=len(examples),
        total_sampled_examples=len(examples),
        total_preference_pairs=0,
        total_curriculum_stages=2,
        overall_quality_mean=8.7,
        execution_time_seconds=1.2,
    )
    stats = DistillationStatistics(
        selection_counts={"selected": 10, "rejected": 0},
        rejection_counts={},
        sampling_statistics={"sampled": 10},
        curriculum_distribution={"easy": 5, "medium": 5},
        preference_counts={},
        teacher_distribution={"teacher_llm_v1": 10},
        difficulty_distribution={"easy": 5, "medium": 5},
        quality_distribution={"8-10": 10},
        total_examples=10,
    )
    manifest = DistillationManifest(
        artifact_id=artifact_id,
        statistics_summary={"total": 10},
        configuration_hash="cfg_hash_mock_123",
        lineage={},
        checksum="chk_mock_123",
    )
    report = DistillationReport(
        selection_summary={"total": 10},
        filtering_summary={"total": 10},
        sampling_summary={"total": 10},
        curriculum_summary={"stages": 2},
        preference_summary={},
        statistics=stats,
        configuration_summary={},
        dataset_summary={},
        benchmark_summary={},
    )
    return DistillationArtifact(
        artifact_id=artifact_id,
        metadata=meta,
        lineage=lineage,
        configuration=cfg,
        summary=summary,
        statistics=stats,
        manifest=manifest,
        dataset=dist_dataset,
        report=report,
    )


@app.command("run")
def run_finetuning(
    artifact_path: str | None = typer.Option(
        None, "--artifact", "-a", help="Path to input DistillationArtifact JSON file"
    ),
    run_name: str = typer.Option(
        "qwen2.5-7b-coaching-qlora", "--run-name", "-n", help="Unique run name identifier"
    ),
    strategy: str = typer.Option(
        "SFTTrainingStrategy",
        "--strategy",
        "-s",
        help="Training strategy (SFTTrainingStrategy, DPOTrainingStrategy, ORPOTrainingStrategy, CurriculumTrainingStrategy, HybridTrainingStrategy)",
    ),
    backend: str = typer.Option(
        "mock",
        "--backend",
        "-b",
        help="Training backend framework (mock, unsloth, axolotl, huggingface, trl)",
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    learning_rate: float = typer.Option(0.0002, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(4, "--batch-size", help="Per device train batch size"),
    output_dir: str = typer.Option(
        "outputs/finetuning", "--output-dir", "-o", help="Target output directory"
    ),
    resume: str | None = typer.Option(
        None, "--resume", "-r", help="Checkpoint directory or step to resume training from"
    ),
) -> None:
    """Execute Fine-Tuning Pipeline to train a model artifact from DistillationArtifact."""
    console.print(
        f"[bold cyan]Launching TradeSense Fine-Tuning Pipeline:[/bold cyan] [green]{run_name}[/green]"
    )

    if artifact_path and Path(artifact_path).exists():
        dist_artifact = FineTuningPipeline.load_distillation_artifact(artifact_path)
    else:
        dist_artifact = _create_mock_distillation_artifact()
        console.print(
            "[yellow]No input artifact path supplied; using deterministic DistillationArtifact payload.[/yellow]"
        )

    backend_cfg = TrainingBackendConfiguration(backend_name=backend)
    config = TrainingConfiguration(
        run_name=run_name,
        strategy_name=strategy,
        backend_config=backend_cfg,
        output_dir=output_dir,
    )
    config.model_config_params.model_copy(
        update={
            "num_epochs": epochs,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
        }
    )

    pipeline = FineTuningPipeline()
    model_artifact = pipeline.run(
        input_data=dist_artifact,
        config=config,
        output_dir=output_dir,
        resume_from_checkpoint=resume,
    )

    console.print(
        f"[bold green]Fine-Tuning completed successfully![/bold green] ModelArtifact ID: [cyan]{model_artifact.artifact_id}[/cyan]"
    )

    table = Table(title="Model Artifact Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Model ID", model_artifact.artifact_id)
    table.add_row("Strategy", model_artifact.metadata.strategy_name)
    table.add_row("Backend", model_artifact.metadata.backend_name)
    table.add_row("Final Loss", str(model_artifact.summary.final_loss))
    table.add_row("Eval Loss", str(model_artifact.summary.eval_loss))
    table.add_row("Total Steps", str(model_artifact.summary.total_steps))
    table.add_row("Best Checkpoint", str(model_artifact.summary.best_checkpoint_id or "N/A"))
    table.add_row("Package Checksum", model_artifact.package.package_checksum[:16] + "...")

    console.print(table)


@app.command("validate")
def validate_artifact(
    artifact_path: str = typer.Argument(
        ..., help="Path to ModelArtifact JSON file or DistillationArtifact JSON file"
    ),
) -> None:
    """Validate schema integrity and completeness of a DistillationArtifact or ModelArtifact."""
    path = Path(artifact_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found at '{artifact_path}'")
        raise typer.Exit(code=1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    validator = FineTuningValidator()
    if "package" in data or "model_id" in data.get("metadata", {}):
        art = ModelArtifact.model_validate(data)
        issues = validator.validate_model_artifact(art)
        if not issues:
            console.print(
                f"[bold green]Validation PASSED:[/bold green] ModelArtifact '{art.artifact_id}' is valid and complete."
            )
        else:
            console.print(
                f"[bold yellow]Validation WARNINGS:[/bold yellow] Found {len(issues)} issues:"
            )
            for iss in issues:
                console.print(f"- [red]{iss}[/red]")
    else:
        dist_art = DistillationArtifact.model_validate(data)
        issues = validator.validate_distillation_artifact(dist_art)
        if not issues:
            console.print(
                f"[bold green]Validation PASSED:[/bold green] DistillationArtifact '{dist_art.artifact_id}' is compatible for fine-tuning."
            )
        else:
            console.print(f"[bold red]Validation FAILED:[/bold red] Found {len(issues)} issues:")
            for iss in issues:
                console.print(f"- [red]{iss}[/red]")


@app.command("report")
def generate_report(
    model_artifact_path: str = typer.Argument(..., help="Path to ModelArtifact JSON file"),
    output_md: str | None = typer.Option(
        None, "--output", "-o", help="Path to save Markdown report"
    ),
) -> None:
    """Generate and display structured training report from ModelArtifact."""
    path = Path(model_artifact_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found at '{model_artifact_path}'")
        raise typer.Exit(code=1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    art = ModelArtifact.model_validate(data)
    reporter = FineTuningReporter()
    md = reporter.render_markdown_report(art.report)

    if output_md:
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md)
        console.print(f"[green]Report saved to '{output_md}'[/green]")
    else:
        console.print(md)


@app.command("export")
def export_artifact(
    model_artifact_path: str = typer.Argument(..., help="Path to ModelArtifact JSON file"),
    output_dir: str = typer.Option(
        "outputs/exports", "--output-dir", "-o", help="Directory for exports"
    ),
    formats: list[str] = typer.Option(
        ["directory", "json", "markdown", "manifest"], "--format", "-f", help="Export target format"
    ),
) -> None:
    """Export ModelArtifact to target release formats."""
    path = Path(model_artifact_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found at '{model_artifact_path}'")
        raise typer.Exit(code=1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    art = ModelArtifact.model_validate(data)
    exporter = ModelExporter()
    descs = exporter.export(art, output_dir, formats)

    console.print(
        f"[bold green]Exported ModelArtifact '{art.artifact_id}' to '{output_dir}':[/bold green]"
    )
    for d in descs:
        console.print(f"- [{d['format']}] [cyan]{d['path']}[/cyan]")


@app.command("checkpoints")
def list_checkpoints(
    run_dir: str = typer.Argument(..., help="Run output directory containing checkpoints/ folder"),
) -> None:
    """Inspect and validate checkpoints in a run output directory."""
    mgr = CheckpointManager(output_dir=run_dir)
    ckpts_dir = Path(run_dir) / "checkpoints"
    if not ckpts_dir.exists():
        console.print(f"[yellow]No checkpoints directory found at '{ckpts_dir}'[/yellow]")
        return

    table = Table(title=f"Checkpoints in {run_dir}")
    table.add_column("Checkpoint ID", style="cyan")
    table.add_column("Path", style="magenta")
    table.add_column("Status", style="green")

    for item in ckpts_dir.iterdir():
        if item.is_dir():
            valid = mgr.validate_checkpoint(str(item))
            table.add_row(item.name, str(item), "Valid" if valid else "Corrupted/Incomplete")

    console.print(table)
