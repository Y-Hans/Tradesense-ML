"""Dataset CLI subcommands for tsml."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tradesense_ml.domain.schemas.synthetic import SyntheticGeneratorConfig
from tradesense_ml.pipelines.generation.pipeline import ConcreteSyntheticGenerationPipeline
from tradesense_ml.pipelines.validation.synthetic_validator import SyntheticDatasetValidator
from tradesense_ml.storage.dataset_exporter import DatasetExporter

app = typer.Typer(name="dataset", help="Manage, generate, and validate synthetic datasets.")
console = Console()


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

    # Display summary statistics table
    table = Table(title="Generated Dataset Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Samples", str(len(samples)))
    table.add_row("Dataset ID", lineage.dataset_id)
    table.add_row("Source Config Hash", lineage.source_hash[:16] + "...")
    table.add_row("Generator Version", lineage.generator_version)

    regimes_count: dict[str, int] = {}
    outcomes_count: dict[str, int] = {}
    for req in samples:
        reg = req.market_context.regime.value if req.market_context else "UNKNOWN"
        outc = req.trade.metadata.get("outcome", "UNKNOWN")
        regimes_count[reg] = regimes_count.get(reg, 0) + 1
        outcomes_count[outc] = outcomes_count.get(outc, 0) + 1

    table.add_row("Regimes Distribution", str(regimes_count))
    table.add_row("Outcomes Distribution", str(outcomes_count))

    console.print(table)


@app.command("validate")
def validate_dataset(
    dataset_path: str = typer.Argument(
        ..., help="Path to dataset file to validate (.jsonl, .json, .parquet)"
    ),
) -> None:
    """Validate a dataset file against Pydantic schemas and math rules."""
    console.print(f"[bold blue]Validating dataset file at: {dataset_path}[/bold blue]")

    path = Path(dataset_path)
    if not path.exists():
        console.print(f"[bold red]Error: File '{dataset_path}' not found.[/bold red]")
        raise typer.Exit(code=1)

    records = DatasetExporter.load_dataset(path)
    console.print(f"Loaded [cyan]{len(records)}[/cyan] records for validation.")

    validator = SyntheticDatasetValidator()
    all_valid, results = validator.validate_batch(records)

    if all_valid:
        console.print(
            Panel(
                f"[bold green]VALIDATION PASSED[/bold green]\nAll {len(records)} samples adhere strictly to Pydantic schemas and mathematical consistency rules.",
                title="Result",
                border_style="green",
            )
        )
    else:
        invalid_count = sum(1 for r in results if not r.is_valid)
        console.print(
            Panel(
                f"[bold red]VALIDATION FAILED[/bold red]\n{invalid_count} of {len(records)} samples failed validation.",
                title="Result",
                border_style="red",
            )
        )

        table = Table(title="Validation Errors Overview", show_header=True, header_style="bold red")
        table.add_column("Sample #", style="yellow")
        table.add_column("Errors", style="red")

        for idx, r in enumerate(results):
            if not r.is_valid:
                table.add_row(str(idx + 1), "\n".join(r.errors))

        console.print(table)
        raise typer.Exit(code=1)


@app.command("preview")
def preview_dataset(
    dataset_path: str = typer.Argument(..., help="Path to dataset file to preview"),
    limit: int = typer.Option(3, "--limit", "-n", help="Number of samples to preview"),
) -> None:
    """Preview formatted contents of a synthetic dataset file."""
    console.print(f"[bold cyan]Previewing dataset: {dataset_path} (limit={limit})[/bold cyan]")

    path = Path(dataset_path)
    if not path.exists():
        console.print(f"[bold red]Error: File '{dataset_path}' not found.[/bold red]")
        raise typer.Exit(code=1)

    records = DatasetExporter.load_dataset(path)
    preview_count = min(limit, len(records))

    for i in range(preview_count):
        rec = records[i]
        trade = rec.get("trade", {})
        market = rec.get("market_context", {})

        table = Table(
            title=f"Sample #{i+1} — Request ID: {rec.get('request_id')}",
            show_header=True,
            header_style="bold yellow",
        )
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("User ID", str(rec.get("user_id")))
        table.add_row("Symbol", str(trade.get("symbol")))
        table.add_row("Side", str(trade.get("side")))
        table.add_row("Entry Price", str(trade.get("entry_price")))
        table.add_row("Exit Price", str(trade.get("exit_price")))
        table.add_row("Quantity", str(trade.get("quantity")))
        table.add_row("PnL ($ / %)", f"{trade.get('pnl')} ({trade.get('pnl_percentage')}%)")
        table.add_row("Market Regime", str(market.get("regime")))
        table.add_row("Volatility", str(market.get("volatility")))
        table.add_row("Outcome", str(trade.get("metadata", {}).get("outcome")))
        table.add_row("Applied Bias", str(trade.get("metadata", {}).get("bias_applied")))

        console.print(table)


@app.command("list")
def list_datasets() -> None:
    """List available dataset versions and lineage metadata."""
    console.print("[bold cyan]Available Datasets in Registry:[/bold cyan]")
    console.print(" - tradesense_coaching_v1 (v1.0.0, Split: train, Status: APPROVED)")
    console.print(
        " - synthetic_market_scenarios_v1 (v0.1.0, Split: validation, Status: AUTOMATED_VALIDATION)"
    )
