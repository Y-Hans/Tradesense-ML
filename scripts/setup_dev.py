"""Development environment setup verification script."""

import importlib.util
import sys

from rich.console import Console

console = Console()


def check_environment() -> None:
    """Verify python version and key packages."""
    console.print(f"[bold green]Python version:[/bold green] {sys.version}")

    packages = ["pydantic", "typer", "hydra", "loguru", "mlflow"]
    missing = []

    for pkg in packages:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)

    if not missing:
        console.print(
            "[green]✓ Core packages (pydantic, typer, hydra, loguru, mlflow) found.[/green]"
        )
    else:
        console.print(f"[red]Missing dependencies: {', '.join(missing)}[/red]")


if __name__ == "__main__":
    check_environment()
