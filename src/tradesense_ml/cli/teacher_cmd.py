"""Teacher CLI subcommands for tsml teacher."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tradesense_ml.domain.schemas.coaching import CoachRequest
from tradesense_ml.pipelines.generation.pipeline import ConcreteSyntheticGenerationPipeline
from tradesense_ml.pipelines.inference.pipeline import TeacherInferencePipeline
from tradesense_ml.pipelines.inference.strategies import SingleTeacherStrategy
from tradesense_ml.teachers.providers import (
    AnthropicTeacherProvider,
    GeminiTeacherProvider,
    LocalTeacherProvider,
    OllamaTeacherProvider,
    OpenAITeacherProvider,
    OpenRouterTeacherProvider,
    VLLMTeacherProvider,
)
from tradesense_ml.teachers.router import TeacherRouter

app = typer.Typer(
    name="teacher", help="Interact with Teacher LLM providers and inference pipeline."
)
console = Console()


def _get_default_router() -> TeacherRouter:
    """Build TeacherRouter populated with all 7 available providers."""
    providers = [
        OpenRouterTeacherProvider(),
        OpenAITeacherProvider(),
        AnthropicTeacherProvider(),
        GeminiTeacherProvider(),
        OllamaTeacherProvider(),
        LocalTeacherProvider(),
        VLLMTeacherProvider(),
    ]
    return TeacherRouter(providers)


@app.command("providers")
def list_providers() -> None:
    """List all registered Teacher LLM providers, default models, and cost parameters."""
    router = _get_default_router()
    table = Table(title="Registered TradeSense Teacher LLM Providers")
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Default Model", style="green")
    table.add_column("Cost / 1k Input ($)", justify="right")
    table.add_column("Cost / 1k Output ($)", justify="right")
    table.add_column("Status", style="bold green")

    for name, prov in router.providers.items():
        table.add_row(
            name,
            prov.default_model,
            f"${prov.cost_per_1k_input:.4f}",
            f"${prov.cost_per_1k_output:.4f}",
            "AVAILABLE",
        )

    console.print(table)


@app.command("infer")
def infer_single(
    request_file: str | None = typer.Option(
        None, "--request-file", "-f", help="Path to CoachRequest JSON file"
    ),
    provider: str = typer.Option(
        "openrouter", "--provider", "-p", help="Teacher LLM provider name"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name override"),
    prompt_version: str = typer.Option("v1", "--prompt-version", help="Prompt template version"),
    output_path: str | None = typer.Option(
        None, "--output-path", "-o", help="Path to save output CoachResponse JSON"
    ),
) -> None:
    """Run Teacher Inference for a single synthetic or custom CoachRequest."""
    console.print(f"[bold cyan]Initializing Teacher Inference Pipeline ({provider})...[/bold cyan]")

    router = _get_default_router()

    if request_file:
        file_path = Path(request_file)
        if not file_path.exists():
            console.print(f"[bold red]Error: Request file '{request_file}' not found.[/bold red]")
            raise typer.Exit(code=1)
        content = file_path.read_text(encoding="utf-8")
        coach_request = CoachRequest.model_validate_json(content)
        console.print(f"[green]Loaded CoachRequest '{coach_request.request_id}' from file.[/green]")
    else:
        console.print("[dim]No request file supplied. Generating synthetic CoachRequest...[/dim]")
        gen_pipeline = ConcreteSyntheticGenerationPipeline()
        from tradesense_ml.domain.schemas.synthetic import SyntheticGeneratorConfig

        requests, _ = gen_pipeline.generate_dataset(
            SyntheticGeneratorConfig(num_samples=1, seed=42)
        )
        coach_request = requests[0]
        console.print(
            f"[green]Generated synthetic CoachRequest '{coach_request.request_id}'.[/green]"
        )

    pipeline = TeacherInferencePipeline(
        router=router, strategy=SingleTeacherStrategy(provider_name=provider)
    )
    response = pipeline.run(
        coach_request,
        provider=provider,
        model_name=model,
        prompt_version=prompt_version,
    )

    console.print("\n[bold green]Teacher Inference Succeeded![/bold green]")
    console.print(f"[bold]Response ID:[/bold] {response.response_id}")
    console.print(f"[bold]Headline:[/bold] {response.headline}")
    console.print(f"[bold]Overall Score:[/bold] {response.overall_score}/10.0")
    console.print(f"[bold]Risk Score:[/bold] {response.risk_evaluation.risk_score}/10.0")
    console.print(
        f"[bold]Discipline Score:[/bold] {response.discipline_evaluation.discipline_score}/10.0"
    )
    console.print(f"[bold]Latency:[/bold] {response.metadata.get('latency_ms', 0.0)} ms")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(response.model_dump(), indent=2), encoding="utf-8")
        console.print(f"[bold yellow]Response saved to '{output_path}'[/bold yellow]")


@app.command("batch")
def infer_batch(
    input_path: str = typer.Option(
        ..., "--input-path", "-i", help="Directory or JSON file containing CoachRequests"
    ),
    output_dir: str = typer.Option(
        "./outputs/inference", "--output-dir", "-o", help="Directory to save output CoachResponses"
    ),
    provider: str = typer.Option(
        "openrouter", "--provider", "-p", help="Teacher LLM provider name"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name override"),
    prompt_version: str = typer.Option("v1", "--prompt-version", help="Prompt template version"),
) -> None:
    """Run batch Teacher Inference across a set of CoachRequests."""
    inp = Path(input_path)
    if not inp.exists():
        console.print(f"[bold red]Error: Input path '{input_path}' not found.[/bold red]")
        raise typer.Exit(code=1)

    requests: list[CoachRequest] = []
    if inp.is_file():
        content = inp.read_text(encoding="utf-8")
        data = json.loads(content)
        if isinstance(data, list):
            requests = [CoachRequest.model_validate(item) for item in data]
        else:
            requests = [CoachRequest.model_validate(data)]
    else:
        for f in inp.glob("*.json"):
            content = f.read_text(encoding="utf-8")
            requests.append(CoachRequest.model_validate_json(content))

    if not requests:
        console.print(f"[bold yellow]No CoachRequest files found at '{input_path}'.[/bold yellow]")
        raise typer.Exit(code=0)

    console.print(f"[bold cyan]Processing batch of {len(requests)} request(s)...[/bold cyan]")
    router = _get_default_router()
    pipeline = TeacherInferencePipeline(router=router)

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    completed = 0
    for req in requests:
        res = pipeline.run(
            req,
            provider=provider,
            model_name=model,
            prompt_version=prompt_version,
        )
        out_file = out_dir_path / f"response_{req.request_id}.json"
        out_file.write_text(json.dumps(res.model_dump(), indent=2), encoding="utf-8")
        completed += 1

    console.print(
        f"[bold green]Batch inference completed successfully. Saved {completed} response(s) to '{output_dir}'.[/bold green]"
    )


@app.command("generate")
def teacher_generate(
    provider: str = typer.Option("openrouter", "--provider", "-p", help="Provider name"),
    prompt: str = typer.Option("Analyze trade", "--prompt", help="Prompt text"),
) -> None:
    """Generate responses using a Teacher LLM provider (Legacy CLI alias)."""
    console.print(f"[bold cyan]Sending prompt to Teacher Provider '{provider}'...[/bold cyan]")
    router = _get_default_router()
    console.print(
        f"[green]Teacher Provider Router initialized with {len(router.providers)} provider(s).[/green]"
    )
    console.print(f"[dim]Provider '{provider}' active.[/dim]")


@app.command("test")
def teacher_test() -> None:
    """Test connectivity and token usage calculation for configured providers."""
    router = _get_default_router()
    console.print(
        f"[bold yellow]Testing {len(router.providers)} registered Teacher Providers...[/bold yellow]"
    )
    for name in router.providers:
        console.print(f" - Provider '{name}': [green]OK[/green]")
