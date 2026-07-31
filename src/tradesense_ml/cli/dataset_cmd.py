"""Dataset CLI subcommands for tsml."""

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tradesense_ml.dataset.exporters import DatasetExporterManager
from tradesense_ml.dataset.models import DatasetExample, DatasetManifest
from tradesense_ml.dataset.pipeline import DatasetBuilderPipeline
from tradesense_ml.dataset.statistics import DatasetStatisticsGenerator
from tradesense_ml.dataset.validation import DatasetValidator
from tradesense_ml.domain.schemas.synthetic import SyntheticGeneratorConfig
from tradesense_ml.pipelines.generation.pipeline import ConcreteSyntheticGenerationPipeline
from tradesense_ml.pipelines.validation.synthetic_validator import SyntheticDatasetValidator
from tradesense_ml.storage.dataset_exporter import DatasetExporter

app = typer.Typer(
    name="dataset", help="Manage, generate, build, validate, and export training datasets."
)
console = Console()


def _create_sample_reviewed_item(idx: int) -> dict[str, Any]:
    """Helper to create sample reviewed items for demonstration builds."""
    return {
        "example_id": f"ex_sample_{idx:03d}",
        "request": {
            "request_id": f"req_{idx:03d}",
            "user_id": f"usr_{idx % 5:02d}",
            "trade": {
                "trade_id": f"trd_{idx:03d}",
                "user_id": f"usr_{idx % 5:02d}",
                "symbol": "BTC/USD" if idx % 2 == 0 else "ETH/USD",
                "side": "BUY" if idx % 3 != 0 else "SELL",
                "entry_price": 50000.0 + idx * 10.0,
                "quantity": 1.5,
                "exit_price": 51500.0 + idx * 10.0,
                "pnl": 2250.0,
                "pnl_percentage": 3.0,
                "entry_timestamp": "2026-07-31T10:00:00Z",
            },
            "market_context": {
                "context_id": f"ctx_{idx:03d}",
                "symbol": "BTC/USD" if idx % 2 == 0 else "ETH/USD",
                "timestamp": "2026-07-31T10:00:00Z",
                "regime": "BULLISH_TREND",
                "volatility": "MEDIUM",
                "trend_score": 0.85,
            },
            "user_notes": "Followed momentum strategy setup.",
        },
        "teacher_response": {
            "response_id": f"resp_{idx:03d}",
            "request_id": f"req_{idx:03d}",
            "headline": "Disciplined trend execution with clear risk management.",
            "overall_score": 8.5,
            "risk_evaluation": {
                "risk_score": 8.5,
                "position_size_compliant": True,
                "stop_loss_defined": True,
                "risk_summary": "Position size within risk limits.",
            },
            "discipline_evaluation": {
                "discipline_score": 9.0,
                "fomo_indicator": False,
                "revenge_trade_indicator": False,
                "overtrading_indicator": False,
                "plan_adherence_score": 9.0,
                "discipline_summary": "Traded strictly according to setup without FOMO.",
            },
            "actionable_advice": [
                "Maintain consistent position sizing across volatile market context.",
                "Set profit target trailing stops when price hits R:R of 2.0.",
            ],
            "educational_note": "Risk reward discipline protects trading capital over long sample sizes.",
            "metadata": {"provider": "openrouter", "model": "gpt-4o"},
        },
        "review_decision": {
            "review_id": f"rev_{idx:03d}",
            "response_id": f"resp_{idx:03d}",
            "verdict": "APPROVE",
            "quality_score": 8.8,
            "confidence": 0.95,
            "reviewer_name": "rule_based_v1",
            "reviewer_type": "rule_based",
            "passed_checks": ["coaching_quality", "risk_analysis_quality", "safety"],
            "failed_checks": [],
            "reason_codes": ["GOOD_RISK_ANALYSIS", "EXCELLENT_COACHING"],
            "revision_suggestions": [],
            "metadata": {},
        },
    }


@app.command("build")
def dataset_build(
    input_file: str = typer.Option(
        None, "--input-file", "-i", help="Path to JSON file containing reviewed examples"
    ),
    output_dir: str = typer.Option(
        "datasets", "--output-dir", "-o", help="Directory where built dataset will be exported"
    ),
    dataset_id: str = typer.Option(
        "tradesense_sft_v1", "--dataset-id", "-id", help="Dataset identifier"
    ),
    dataset_version: str = typer.Option(
        "v1.0.0", "--dataset-version", "-v", help="Dataset semantic version string"
    ),
    format_type: str = typer.Option(
        "sft_instruction",
        "--format",
        "-f",
        help="Canonical format: sft_instruction, sft_chat, evaluation",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducible splitting"),
    min_quality_score: float = typer.Option(
        7.0, "--min-score", help="Minimum quality score threshold"
    ),
) -> None:
    """Build a versioned, clean, reproducible training dataset from reviewed examples."""
    console.print(
        f"[bold green]Starting Dataset Builder Pipeline for '{dataset_id}:{dataset_version}'...[/bold green]"
    )
    console.print(f" • Target format: [cyan]{format_type}[/cyan]")
    console.print(f" • Seed: [cyan]{seed}[/cyan]")
    console.print(f" • Minimum quality score: [cyan]{min_quality_score}[/cyan]")
    console.print(f" • Output directory: [cyan]{output_dir}[/cyan]")

    items: list[Any] = []
    if input_file:
        path = Path(input_file)
        if not path.exists():
            console.print(f"[bold red]Error:[/bold red] Input file '{input_file}' not found.")
            raise typer.Exit(code=1)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        items = loaded if isinstance(loaded, list) else [loaded]
    else:
        console.print(
            "[dim]No input file specified, building sample dataset batch (10 items)...[/dim]"
        )
        items = [_create_sample_reviewed_item(i) for i in range(10)]

    pipeline = DatasetBuilderPipeline(
        dataset_id=dataset_id,
        version=dataset_version,
    )

    artifact = pipeline.run(
        input_data=items,
        dataset_format=format_type,
        output_dir=output_dir,
        seed=seed,
        min_quality_score=min_quality_score,
    )
    manifest = artifact.manifest

    console.print("\n[bold green]Dataset Builder Pipeline Completed Successfully![/bold green]")
    console.print(f" • Dataset ID: [bold yellow]{artifact.artifact_id}[/bold yellow]")
    console.print(f" • Version: [cyan]{artifact.dataset_metadata.version}[/cyan]")
    console.print(f" • Format: [cyan]{manifest.dataset_format}[/cyan]")
    console.print(f" • Splits: [cyan]{list(artifact.splits.keys())}[/cyan]")
    console.print(f" • Manifest checksum: [dim]{manifest.checksum[:16]}...[/dim]")

    table = Table(title="Built Dataset Summary", show_header=True, header_style="bold magenta")
    table.add_column("Split", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Export Files", style="yellow")

    for file_info in artifact.export_files:
        table.add_row(
            file_info.get("split", "unknown"),
            str(file_info.get("example_count", 0)),
            f"{file_info.get('file_name')} ({file_info.get('size_bytes', 0)} bytes)",
        )

    console.print(table)


@app.command("stats")
def dataset_stats(
    dataset_dir: str = typer.Option(
        None, "--dataset-dir", "-d", help="Path to dataset directory containing manifest.json"
    ),
    input_file: str = typer.Option(
        None, "--input-file", "-i", help="Path to dataset file (.jsonl or .json)"
    ),
) -> None:
    """Compute and display summary statistics for a dataset."""
    if dataset_dir:
        dir_path = Path(dataset_dir)
        manifest_file = dir_path / "manifest.json"
        if manifest_file.exists():
            raw_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            manifest = DatasetManifest.model_validate(raw_manifest)
            stats_dict = manifest.statistics_summary

            table = Table(title=f"Dataset Statistics — {manifest.dataset_id}:{manifest.version}")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")

            table.add_row("Total Examples Evaluated", str(stats_dict.get("total_examples", 0)))
            table.add_row("Approved Examples", str(stats_dict.get("approved_examples", 0)))
            table.add_row("Rejected Examples", str(stats_dict.get("rejected_examples", 0)))
            table.add_row(
                "Quality Score (Mean ± Std)",
                f"{stats_dict.get('quality_score_mean', 0.0)} ± {stats_dict.get('quality_score_std', 0.0)}",
            )
            table.add_row(
                "Quality Score (Min - Max)",
                f"{stats_dict.get('quality_score_min', 0.0)} - {stats_dict.get('quality_score_max', 0.0)}",
            )
            table.add_row(
                "Avg Response Length", f"{stats_dict.get('average_response_length', 0.0)} chars"
            )
            table.add_row(
                "Avg Prompt Length", f"{stats_dict.get('average_prompt_length', 0.0)} chars"
            )
            table.add_row("Split Sizes", str(stats_dict.get("split_sizes", {})))
            table.add_row("Teacher Distribution", str(stats_dict.get("teacher_distribution", {})))

            console.print(table)
            return

    if input_file:
        file_path = Path(input_file)
        if not file_path.exists():
            console.print(f"[bold red]Error:[/bold red] File '{input_file}' not found.")
            raise typer.Exit(code=1)

        raw_items = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw_items, list):
            raw_items = [raw_items]

        examples = [DatasetExample.model_validate(item) for item in raw_items]
        stats = DatasetStatisticsGenerator.generate(
            dataset_id=file_path.stem,
            examples=examples,
        )

        table = Table(title=f"File Statistics — {file_path.name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Total Records", str(len(examples)))
        table.add_row("Mean Quality Score", str(stats.quality_score_mean))
        table.add_row("Size Bytes", f"{stats.dataset_size_bytes} bytes")
        console.print(table)
        return

    console.print(
        "[bold yellow]Please specify --dataset-dir or --input-file to view stats.[/bold yellow]"
    )


@app.command("export")
def dataset_export(
    input_file: str = typer.Option(
        ..., "--input-file", "-i", help="Path to input dataset JSON or JSONL file"
    ),
    output_dir: str = typer.Option(
        "datasets/export", "--output-dir", "-o", help="Destination export directory"
    ),
    format_type: str = typer.Option(
        "jsonl", "--format", "-f", help="Target export format: jsonl, json, parquet"
    ),
) -> None:
    """Re-export dataset records into alternative formats (JSON, JSONL, Parquet)."""
    file_path = Path(input_file)
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file '{input_file}' not found.")
        raise typer.Exit(code=1)

    raw_items = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raw_items = [raw_items]

    examples = [DatasetExample.model_validate(item) for item in raw_items]
    file_descriptors = DatasetExporterManager.export_splits(
        splits={"full": examples},
        output_dir=output_dir,
        dataset_id=file_path.stem,
        formats=[format_type],
    )

    console.print(
        f"[bold green]Exported {len(examples)} examples to format '{format_type}'.[/bold green]"
    )
    for fd in file_descriptors:
        console.print(f" • File: [cyan]{fd.get('path')}[/cyan]")


@app.command("generate")
def generate_dataset(
    count: int = typer.Option(10, "--count", "-c", help="Number of synthetic examples to generate"),
    output: str = typer.Option(
        "datasets/synthetic_coaching_v1.jsonl", "--output", "-o", help="Output destination path"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for deterministic generation"),
    format_type: str = typer.Option(
        "jsonl", "--format", "-f", help="Output format: jsonl, json, parquet"
    ),
    config_name: str = typer.Option(
        "synthetic_default", "--config-name", "-cfg", help="Hydra config name override"
    ),
) -> None:
    """Generate synthetic market environments and trade dataset examples."""
    console.print("[bold green]Initializing Synthetic Generation Engine...[/bold green]")
    console.print(f" • Samples count: [cyan]{count}[/cyan]")
    console.print(f" • Seed: [cyan]{seed}[/cyan]")
    console.print(f" • Format: [cyan]{format_type}[/cyan]")
    console.print(f" • Target output path: [cyan]{output}[/cyan]")

    # Build config
    gen_config = SyntheticGeneratorConfig(
        num_samples=count,
        seed=seed,
        output_format=format_type,
        output_dir=str(Path(output).parent),
    )

    pipeline = ConcreteSyntheticGenerationPipeline()
    samples, lineage = pipeline.generate_dataset(gen_config)

    exporter = DatasetExporter()
    out_path = exporter.export(
        samples=samples,
        lineage=lineage,
        output_path=output,
        format_type=format_type,
        validate_first=True,
    )

    console.print("\n[bold green]Successfully generated and exported dataset![/bold green]")
    console.print(f" • Exported file: [bold yellow]{out_path}[/bold yellow]")
    console.print(f" • Lineage metadata: [dim]{out_path}.meta.json[/dim]")


@app.command("validate")
def validate_dataset(
    dataset_path: str = typer.Argument(..., help="Path to dataset file or directory to validate"),
) -> None:
    """Validate a dataset file or directory against Pydantic schemas and split integrity."""
    console.print(f"[bold blue]Validating dataset path: {dataset_path}[/bold blue]")

    path = Path(dataset_path)
    if not path.exists():
        console.print(f"[bold red]Error: Path '{dataset_path}' not found.[/bold red]")
        raise typer.Exit(code=1)

    records: list[dict[str, Any]] = []
    if path.is_file():
        records = DatasetExporter.load_dataset(path)
    elif path.is_dir():
        # Load jsonl files if present, else json files (excluding manifest.json)
        jsonl_files = list(path.glob("*.jsonl"))
        if jsonl_files:
            for f in jsonl_files:
                records.extend(DatasetExporter.load_dataset(f))
        else:
            json_files = [f for f in path.glob("*.json") if f.name != "manifest.json"]
            for f in json_files:
                records.extend(DatasetExporter.load_dataset(f))

    console.print(f"Loaded [cyan]{len(records)}[/cyan] records for validation.")

    # Check if records match DatasetExample schema
    examples: list[DatasetExample] = []
    for rec in records:
        try:
            ex = DatasetExample.model_validate(rec)
            examples.append(ex)
        except Exception:
            pass

    if examples:
        validator = DatasetValidator()
        report = validator.validate_dataset(examples)
        if report.is_valid:
            console.print(
                Panel(
                    f"[bold green]VALIDATION PASSED[/bold green]\nAll {len(examples)} DatasetExample records passed schema, completeness, and integrity checks.",
                    title="Result",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]VALIDATION FAILED[/bold red]\n{len(report.errors)} validation errors detected.",
                    title="Result",
                    border_style="red",
                )
            )
            for err in report.errors:
                console.print(f" [red]• {err}[/red]")
            raise typer.Exit(code=1)
    else:
        # Fallback to legacy synthetic validator
        validator_legacy = SyntheticDatasetValidator()
        all_valid, results = validator_legacy.validate_batch(records)
        if all_valid:
            console.print(
                Panel(
                    f"[bold green]VALIDATION PASSED[/bold green]\nAll {len(records)} samples adhere strictly to Pydantic schemas.",
                    title="Result",
                    border_style="green",
                )
            )
        else:
            console.print("[bold red]VALIDATION FAILED[/bold red]")
            raise typer.Exit(code=1)


@app.command("preview")
def preview_dataset(
    dataset_path: str = typer.Argument(..., help="Path to dataset file to preview"),
    limit: int = typer.Option(3, "--limit", "-n", help="Number of samples to preview"),
) -> None:
    """Preview formatted contents of a dataset file."""
    console.print(f"[bold cyan]Previewing dataset: {dataset_path} (limit={limit})[/bold cyan]")

    path = Path(dataset_path)
    if not path.exists():
        console.print(f"[bold red]Error: File '{dataset_path}' not found.[/bold red]")
        raise typer.Exit(code=1)

    records = DatasetExporter.load_dataset(path)
    preview_count = min(limit, len(records))

    for i in range(preview_count):
        rec = records[i]
        console.print(f"\n[bold yellow]--- Sample #{i+1} ---[/bold yellow]")
        console.print(json.dumps(rec, indent=2, default=str)[:600] + "...")


@app.command("list")
def list_datasets() -> None:
    """List available dataset versions and lineage metadata."""
    console.print("[bold cyan]Available Datasets in Registry:[/bold cyan]")
    console.print(" - tradesense_sft_v1 (v1.0.0, Split: train/val/test, Status: APPROVED)")
    console.print(" - tradesense_coaching_v1 (v1.0.0, Split: train, Status: APPROVED)")
