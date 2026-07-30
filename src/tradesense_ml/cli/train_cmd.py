"""Train CLI subcommands for tsml."""

import typer
from rich.console import Console

app = typer.Typer(name="train", help="Execute student model fine-tuning jobs.")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    config: str = typer.Option(
        "configs/training/default.yaml", "--config", "-c", help="Hydra config file"
    ),
    strategy: str = typer.Option(
        "qlora", "--strategy", "-s", help="Training strategy: lora, qlora, full"
    ),
) -> None:
    """Launch fine-tuning process for student models."""
    console.print(
        f"[bold red]Initiating Student Model Fine-Tuning ({strategy.upper()})...[/bold red]"
    )
    console.print(f" Loading Hydra configuration: {config}")
    console.print("[dim]Training pipeline scaffold executed (Scaffold / Not implemented)[/dim]")
