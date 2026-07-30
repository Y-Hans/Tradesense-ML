"""Export CLI subcommands for tsml."""

import typer
from rich.console import Console

app = typer.Typer(name="export", help="Export models for serving & Flutter backend integration.")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    model_id: str = typer.Option(
        "tradesense-qwen2.5-7b-v1", "--model-id", "-m", help="Target model ID"
    ),
    format_type: str = typer.Option(
        "gguf", "--format", "-f", help="Export format: gguf, onnx, safetensors"
    ),
) -> None:
    """Export model weights to target deployment format."""
    console.print(
        f"[bold green]Exporting model '{model_id}' to format '{format_type}'...[/bold green]"
    )
    console.print("[dim]Model export completed (Scaffold / Not implemented)[/dim]")
