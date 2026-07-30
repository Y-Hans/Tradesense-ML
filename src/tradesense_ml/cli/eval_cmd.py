"""Evaluate CLI subcommands for tsml."""

import typer
from rich.console import Console

app = typer.Typer(name="evaluate", help="Evaluate model predictions against rubrics.")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    model_id: str = typer.Option(
        "tradesense-student-v1", "--model-id", "-m", help="Target model ID"
    ),
    rubric: str = typer.Option(
        "risk_discipline_v1", "--rubric", "-r", help="Evaluation rubric asset"
    ),
) -> None:
    """Evaluate model output quality against rubric."""
    console.print(f"[bold blue]Evaluating model '{model_id}' with rubric '{rubric}'...[/bold blue]")
    console.print("[dim]Evaluation pipeline complete (Scaffold / Not implemented)[/dim]")
