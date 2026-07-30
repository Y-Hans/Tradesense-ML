"""Serve CLI subcommands for tsml."""

import typer
from rich.console import Console

app = typer.Typer(name="serve", help="Launch inference serving server.")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    port: int = typer.Option(8000, "--port", "-p", help="HTTP server port"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host address"),
) -> None:
    """Start local inference serving endpoint."""
    console.print(
        f"[bold cyan]Starting TradeSense ML Serving API on http://{host}:{port}...[/bold cyan]"
    )
    console.print("[dim]Server initialized (Scaffold / Not implemented)[/dim]")
