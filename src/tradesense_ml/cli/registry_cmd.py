"""Registry CLI subcommands for tsml."""

import typer
from rich.console import Console

app = typer.Typer(name="registry", help="Query and manage the Model Registry.")
console = Console()


@app.command("list")
def list_registry() -> None:
    """List all registered models and deployment stages."""
    console.print("[bold cyan]Model Registry Contents:[/bold cyan]")
    console.print(" - tradesense-qwen2.5-7b-v1 (Stage: PRODUCTION, Dataset: v1.0.0, Score: 8.9/10)")
    console.print(
        " - tradesense-llama3-8b-v1 (Stage: EXPERIMENTAL, Dataset: v1.1.0, Score: 8.2/10)"
    )


@app.command("register")
def register_model(
    model_id: str = typer.Option(..., "--model-id", "-m", help="Model ID"),
    checkpoint_path: str = typer.Option(..., "--checkpoint", "-c", help="Checkpoint directory"),
) -> None:
    """Register a new model checkpoint in the registry."""
    console.print(
        f"[bold green]Registering model '{model_id}' from '{checkpoint_path}'...[/bold green]"
    )
    console.print("[dim]Model registered in catalog (Scaffold / Not implemented)[/dim]")
