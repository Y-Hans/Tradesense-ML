"""Review CLI subcommands for tsml."""

import typer
from rich.console import Console

app = typer.Typer(name="review", help="Manage multi-stage dataset review pipeline queues.")
console = Console()


@app.command("start")
def review_start(
    dataset_path: str = typer.Argument(..., help="Path to raw dataset for review processing"),
) -> None:
    """Start automated and AI review pipeline on a dataset batch."""
    console.print(
        f"[bold magenta]Starting 4-stage Review Pipeline for: {dataset_path}[/bold magenta]"
    )
    console.print(" Stage 1: Automated Validation [green]PASSED[/green]")
    console.print(" Stage 2: AI Teacher Consensus Review [yellow]IN_PROGRESS[/yellow]")
    console.print("[dim]Review pipeline queued (Scaffold / Not implemented)[/dim]")


@app.command("queue")
def review_queue() -> None:
    """Display items currently waiting in the Human Review queue."""
    console.print("[bold yellow]Human Review Queue Summary:[/bold yellow]")
    console.print(" Total Pending Items: 0")
