"""Main CLI entrypoint for TradeSense ML (`tsml`)."""

import typer
from rich.console import Console

from tradesense_ml import __version__
from tradesense_ml.cli import (
    benchmark_cmd,
    dataset_cmd,
    distillation_cmd,
    eval_cmd,
    export_cmd,
    registry_cmd,
    review_cmd,
    serve_cmd,
    teacher_cmd,
    train_cmd,
)

app = typer.Typer(
    name="tsml",
    help="TradeSense ML — AI Factory & Research Platform CLI",
    add_completion=False,
)

console = Console()

# Register sub-commands
app.add_typer(dataset_cmd.app, name="dataset")
app.add_typer(teacher_cmd.app, name="teacher")
app.add_typer(review_cmd.app, name="review")
app.add_typer(train_cmd.app, name="train")
app.add_typer(eval_cmd.app, name="evaluate")
app.add_typer(benchmark_cmd.app, name="benchmark")
app.add_typer(distillation_cmd.app, name="distillation")
app.add_typer(registry_cmd.app, name="registry")
app.add_typer(export_cmd.app, name="export")
app.add_typer(serve_cmd.app, name="serve")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show TradeSense ML version"),
) -> None:
    """Main CLI callback."""
    if version:
        console.print(
            f"[bold cyan]TradeSense ML CLI version:[/bold cyan] [green]{__version__}[/green]"
        )
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print("[bold yellow]TradeSense ML Research Platform CLI[/bold yellow]")
        console.print("Run [bold cyan]tsml --help[/bold cyan] for available command groups.")


if __name__ == "__main__":
    app()
