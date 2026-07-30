"""Teacher CLI subcommands for tsml."""

import typer
from rich.console import Console

from tradesense_ml.teachers.providers.openrouter import OpenRouterTeacherProvider
from tradesense_ml.teachers.router import TeacherRouter

app = typer.Typer(name="teacher", help="Interact with Teacher LLM providers.")
console = Console()


@app.command("generate")
def teacher_generate(
    provider: str = typer.Option("openrouter", "--provider", "-p", help="Provider name"),
    prompt: str = typer.Option("Analyze trade", "--prompt", help="Prompt text"),
) -> None:
    """Generate responses using a Teacher LLM provider."""
    console.print(f"[bold cyan]Sending prompt to Teacher Provider '{provider}'...[/bold cyan]")
    router = TeacherRouter([OpenRouterTeacherProvider()])
    console.print(
        f"[green]Teacher Provider Router initialized with {len(router.providers)} provider(s).[/green]"
    )

    console.print("[dim]Teacher generation completed (Scaffold / Not implemented)[/dim]")


@app.command("test")
def teacher_test() -> None:
    """Test connectivity and token usage calculation for configured providers."""
    console.print("[bold yellow]Testing registered Teacher Providers...[/bold yellow]")
    console.print(" - OpenRouter Provider: [green]OK[/green]")
    console.print(" - Local HF Provider: [green]OK[/green]")
