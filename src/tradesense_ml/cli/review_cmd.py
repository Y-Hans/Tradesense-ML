"""Review CLI subcommands for tsml."""

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from tradesense_ml.domain.schemas.coaching import CoachResponse
from tradesense_ml.domain.schemas.review import ReviewDecision, ReviewVerdict
from tradesense_ml.review.pipeline import ReviewPipeline
from tradesense_ml.review.reviewers.ai_mock import (
    ClaudeReviewer,
    ConsensusReviewer,
    GeminiReviewer,
    GPTReviewer,
    HumanReviewer,
)
from tradesense_ml.review.reviewers.rule_based import RuleBasedReviewer
from tradesense_ml.review.strategies import SingleReviewerStrategy

app = typer.Typer(name="review", help="Execute and manage CoachResponse review pipeline tasks.")
console = Console()


def _get_reviewer(reviewer_name: str) -> Any:
    """Helper to instantiate reviewer by name."""
    name = reviewer_name.lower().strip()
    if name in ["rule_based", "rule", "default"]:
        return RuleBasedReviewer()
    elif "gpt" in name:
        return GPTReviewer()
    elif "claude" in name:
        return ClaudeReviewer()
    elif "gemini" in name:
        return GeminiReviewer()
    elif "consensus" in name:
        return ConsensusReviewer()
    elif "human" in name:
        return HumanReviewer()
    else:
        return RuleBasedReviewer(reviewer_name=reviewer_name)


@app.command("run")
def review_run(
    input_file: str = typer.Option(
        None, "--input-file", "-i", help="Path to JSON file containing CoachResponse"
    ),
    json_str: str = typer.Option(None, "--json-str", "-j", help="JSON string of CoachResponse"),
    output_file: str = typer.Option(
        None, "--output-file", "-o", help="Optional path to save ReviewDecision JSON"
    ),
    reviewer: str = typer.Option("rule_based", "--reviewer", "-r", help="Reviewer implementation"),
    strategy: str = typer.Option("single", "--strategy", "-s", help="Review strategy"),
    approval_threshold: float = typer.Option(
        7.0, "--approval-threshold", help="Approval quality score threshold (0-10)"
    ),
    revision_threshold: float = typer.Option(
        4.0, "--revision-threshold", help="Revision quality score threshold (0-10)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON instead of table"),
) -> None:
    """Evaluate a single CoachResponse and produce a ReviewDecision."""
    raw_dict: dict[str, Any] | None = None

    if input_file:
        path = Path(input_file)
        if not path.exists():
            console.print(f"[bold red]Error:[/bold red] Input file '{input_file}' not found.")
            raise typer.Exit(code=1)
        raw_dict = json.loads(path.read_text(encoding="utf-8"))
    elif json_str:
        raw_dict = json.loads(json_str)
    else:
        # Generate sample CoachResponse if no input provided
        console.print("[dim]No input provided, evaluating sample CoachResponse...[/dim]")
        raw_dict = {
            "response_id": "resp_sample_cli_01",
            "request_id": "req_sample_01",
            "headline": "Solid trade execution with well-managed risk reward ratio.",
            "overall_score": 8.0,
            "risk_evaluation": {
                "risk_score": 8.0,
                "position_size_compliant": True,
                "stop_loss_defined": True,
                "risk_summary": "Position size within risk limits and stop loss clearly defined.",
            },
            "discipline_evaluation": {
                "discipline_score": 8.5,
                "fomo_indicator": False,
                "revenge_trade_indicator": False,
                "overtrading_indicator": False,
                "plan_adherence_score": 8.5,
                "discipline_summary": "Traded strictly according to setup without FOMO.",
            },
            "actionable_advice": [
                "Maintain consistent position sizing across volatile market context.",
                "Set profit target trailing stops when price hits R:R of 2.0.",
            ],
            "educational_note": "Risk reward discipline protects trading capital over long sample sizes.",
            "metadata": {"sample": True},
        }

    assert raw_dict is not None
    coach_resp = CoachResponse.model_validate(raw_dict)
    reviewer_inst = _get_reviewer(reviewer)
    pipeline = ReviewPipeline(
        reviewer=reviewer_inst,
        strategy=SingleReviewerStrategy(),
    )

    decision = pipeline.run(
        coach_resp,
        approval_threshold=approval_threshold,
        revision_threshold=revision_threshold,
    )

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(decision.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Saved ReviewDecision to {output_file}[/green]")

    if json_output:
        console.print(decision.model_dump_json(indent=2))
        return

    # Rich table output
    table = Table(title=f"Review Decision — {decision.review_id}")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    verdict_style = (
        "green"
        if decision.verdict == ReviewVerdict.APPROVE
        else ("yellow" if decision.verdict == ReviewVerdict.NEEDS_REVISION else "red")
    )

    table.add_row("Response ID", decision.response_id)
    table.add_row("Verdict", f"[{verdict_style}]{decision.verdict.value}[/{verdict_style}]")
    table.add_row("Quality Score", f"{decision.quality_score} / 10.0")
    table.add_row("Reviewer", f"{decision.reviewer_name} ({decision.reviewer_type})")
    table.add_row("Passed Checks", ", ".join(decision.passed_checks) or "None")
    table.add_row("Failed Checks", ", ".join(decision.failed_checks) or "None")
    table.add_row("Reason Codes", ", ".join([rc.value for rc in decision.reason_codes]))
    table.add_row("Suggestions", "\n".join(decision.revision_suggestions) or "None")

    console.print(table)


@app.command("batch")
def review_batch(
    input_file: str = typer.Option(
        ..., "--input-file", "-i", help="Path to JSON file containing array of CoachResponses"
    ),
    output_file: str = typer.Option(
        "outputs/reviews/batch_results.json",
        "--output-file",
        "-o",
        help="Path to save output decisions JSON",
    ),
    reviewer: str = typer.Option("rule_based", "--reviewer", "-r", help="Reviewer implementation"),
    approval_threshold: float = typer.Option(
        7.0, "--approval-threshold", help="Approval threshold"
    ),
    revision_threshold: float = typer.Option(
        4.0, "--revision-threshold", help="Revision threshold"
    ),
) -> None:
    """Evaluate a batch array of CoachResponses and produce ReviewDecisions."""
    path = Path(input_file)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Batch input file '{input_file}' not found.")
        raise typer.Exit(code=1)

    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        console.print("[bold red]Error:[/bold red] Batch input file must contain a JSON array.")
        raise typer.Exit(code=1)

    reviewer_inst = _get_reviewer(reviewer)
    pipeline = ReviewPipeline(reviewer=reviewer_inst)

    decisions: list[ReviewDecision] = []
    for raw in items:
        resp = CoachResponse.model_validate(raw)
        dec = pipeline.run(
            resp,
            approval_threshold=approval_threshold,
            revision_threshold=revision_threshold,
        )
        decisions.append(dec)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([d.model_dump(mode="json") for d in decisions], indent=2, default=str),
        encoding="utf-8",
    )

    console.print(f"[bold green]Processed {len(decisions)} responses in batch.[/bold green]")
    console.print(f"Results saved to: [cyan]{output_file}[/cyan]")


@app.command("report")
def review_report(
    input_file: str = typer.Option(
        "outputs/reviews/batch_results.json",
        "--input-file",
        "-i",
        help="Path to JSON file containing ReviewDecisions array",
    ),
) -> None:
    """Generate a summary report from a batch of ReviewDecisions."""
    path = Path(input_file)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Decision file '{input_file}' not found.")
        raise typer.Exit(code=1)

    raw_items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        console.print("[bold red]Error:[/bold red] File must contain a JSON array of decisions.")
        raise typer.Exit(code=1)

    decisions = [ReviewDecision.model_validate(item) for item in raw_items]
    total = len(decisions)
    if total == 0:
        console.print("[yellow]No decisions found in report file.[/yellow]")
        return

    approved = sum(1 for d in decisions if d.verdict == ReviewVerdict.APPROVE)
    revised = sum(1 for d in decisions if d.verdict == ReviewVerdict.NEEDS_REVISION)
    rejected = sum(1 for d in decisions if d.verdict == ReviewVerdict.REJECT)
    avg_score = sum(d.quality_score for d in decisions) / float(total)

    failed_check_counts: dict[str, int] = {}
    reason_code_counts: dict[str, int] = {}

    for d in decisions:
        for fc in d.failed_checks:
            failed_check_counts[fc] = failed_check_counts.get(fc, 0) + 1
        for rc in d.reason_codes:
            code_str = rc.value if hasattr(rc, "value") else str(rc)
            reason_code_counts[code_str] = reason_code_counts.get(code_str, 0) + 1

    table = Table(title=f"Review Pipeline Quality Summary ({total} Total Items)")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold magenta")

    table.add_row("Total Evaluated", str(total))
    table.add_row("Approved", f"[green]{approved}[/green] ({round(approved/total*100, 1)}%)")
    table.add_row("Needs Revision", f"[yellow]{revised}[/yellow] ({round(revised/total*100, 1)}%)")
    table.add_row("Rejected", f"[red]{rejected}[/red] ({round(rejected/total*100, 1)}%)")
    table.add_row("Average Quality Score", f"{round(avg_score, 2)} / 10.0")

    top_failed = ", ".join(
        [
            f"{k}: {v}"
            for k, v in sorted(failed_check_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
    )
    table.add_row("Top Failed Checks", top_failed or "None")

    top_reasons = ", ".join(
        [
            f"{k}: {v}"
            for k, v in sorted(reason_code_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
    )
    table.add_row("Top Reason Codes", top_reasons or "None")

    console.print(table)


@app.command("start")
def review_start(
    dataset_path: str = typer.Argument(..., help="Path to raw dataset for review processing"),
) -> None:
    """Legacy endpoint: Start automated and AI review pipeline on a dataset batch."""
    console.print(f"[bold magenta]Starting Review Pipeline for: {dataset_path}[/bold magenta]")
    console.print(" Standard Review Pipeline initialized.")


@app.command("queue")
def review_queue() -> None:
    """Legacy endpoint: Display items currently waiting in the Human Review queue."""
    console.print("[bold yellow]Human Review Queue Summary:[/bold yellow]")
    console.print(" Total Pending Items: 0")
