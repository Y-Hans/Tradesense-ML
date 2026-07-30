"""Benchmark CLI subcommands for tsml."""

import typer
from rich.console import Console

from tradesense_ml.assets_manager.manager import AssetManager

app = typer.Typer(name="benchmark", help="Run versioned benchmark suites.")
console = Console()


@app.command("run")
def run_benchmark(
    benchmark_id: str = typer.Option(
        "coaching_benchmark_v1", "--benchmark", "-b", help="Benchmark asset name"
    ),
    version: str = typer.Option("v1", "--version", "-v", help="Benchmark version"),
) -> None:
    """Run reproducible benchmark suite against student or teacher models."""
    console.print(f"[bold green]Running Benchmark Suite: {benchmark_id} ({version})[/bold green]")
    manager = AssetManager()
    try:
        data = manager.get_benchmark(benchmark_id, version=version)
        console.print(f" Loaded benchmark definition: [bold]{data.get('name')}[/bold]")
        console.print(f" Target metrics: {', '.join(data.get('target_metrics', []))}")
    except Exception as e:
        console.print(f"[red]Error loading benchmark asset: {e}[/red]")
    console.print("[dim]Benchmark execution completed (Scaffold / Not implemented)[/dim]")


@app.command("list")
def list_benchmarks() -> None:
    """List available benchmark suites."""
    console.print("[bold cyan]Available Benchmark Assets:[/bold cyan]")
    console.print(
        " - coaching_benchmark_v1 (v1.0.0, Metrics: JSON validity, Reason codes, Risk/Discipline scores)"
    )
