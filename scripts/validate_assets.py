"""Utility script to validate all files in assets/ directory."""

from rich.console import Console

from tradesense_ml.assets_manager.manager import AssetManager

console = Console()


def validate_all_assets() -> None:
    """Load and validate prompts, rubrics, and benchmarks."""
    console.print("[bold cyan]Validating repository assets...[/bold cyan]")
    manager = AssetManager()

    prompt = manager.get_prompt("system_coach", version="v1")
    console.print(f" Prompt 'system_coach' (v1): [green]OK[/green] ({len(prompt)} chars)")

    rubric = manager.get_rubric("risk_discipline_rubric", version="v1")
    console.print(
        f" Rubric 'risk_discipline_rubric' (v1): [green]OK[/green] ({len(rubric.criteria)} criteria)"
    )

    benchmark = manager.get_benchmark("coaching_benchmark_v1", version="v1")
    console.print(
        f" Benchmark 'coaching_benchmark_v1' (v1): [green]OK[/green] ({benchmark.get('name')})"
    )

    console.print("[bold green]All repository assets validated successfully![/bold green]")


if __name__ == "__main__":
    validate_all_assets()
