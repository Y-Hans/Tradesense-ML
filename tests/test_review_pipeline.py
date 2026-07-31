"""Comprehensive unit tests for the Review Pipeline, ReviewResult, DecisionEngine, Reviewers, Strategies, and CLI subcommands."""

import json
from pathlib import Path

from typer.testing import CliRunner

from tradesense_ml.cli.main import app as cli_app
from tradesense_ml.config.settings import load_hydra_config
from tradesense_ml.domain.schemas.coaching import CoachResponse
from tradesense_ml.domain.schemas.evaluations import DisciplineEvaluation, RiskEvaluation
from tradesense_ml.domain.schemas.review import (
    ReasonCode,
    ReviewDecision,
    ReviewResult,
    ReviewVerdict,
)
from tradesense_ml.review.codes import RevisionSuggestionGenerator
from tradesense_ml.review.criteria import ReviewCriteriaSuite
from tradesense_ml.review.decision_engine import DecisionEngine
from tradesense_ml.review.pipeline import ReviewPipeline
from tradesense_ml.review.reviewers.ai_mock import (
    ClaudeReviewer,
    ConsensusReviewer,
    GeminiReviewer,
    GPTReviewer,
    HumanReviewer,
)
from tradesense_ml.review.reviewers.rule_based import RuleBasedReviewer
from tradesense_ml.review.scoring import QualityScorer
from tradesense_ml.review.strategies import (
    ConsensusStrategy,
    DebateStrategy,
    MultiReviewerStrategy,
    SingleReviewerStrategy,
)

runner = CliRunner()


def _create_sample_coach_response(
    response_id: str = "resp_test_01",
    headline: str = "Solid trade execution with well-managed risk reward ratio.",
    overall_score: float = 8.5,
    risk_score: float = 8.0,
    risk_summary: str = "Position size compliant and stop loss defined.",
    discipline_score: float = 9.0,
    discipline_summary: str = "Traded strictly according to setup without FOMO.",
    actionable_advice: list[str] | None = None,
    educational_note: str = "Risk reward discipline protects trading capital over long sample sizes.",
) -> CoachResponse:
    """Helper to create a CoachResponse domain object."""
    if actionable_advice is None:
        actionable_advice = [
            "Maintain consistent position sizing across volatile market context.",
            "Set profit target trailing stops when price hits R:R of 2.0.",
        ]

    return CoachResponse(
        response_id=response_id,
        request_id="req_test_01",
        headline=headline,
        overall_score=overall_score,
        risk_evaluation=RiskEvaluation(
            risk_score=risk_score,
            position_size_compliant=True,
            stop_loss_defined=True,
            risk_summary=risk_summary,
        ),
        discipline_evaluation=DisciplineEvaluation(
            discipline_score=discipline_score,
            fomo_indicator=False,
            revenge_trade_indicator=False,
            overtrading_indicator=False,
            plan_adherence_score=discipline_score,
            discipline_summary=discipline_summary,
        ),
        actionable_advice=actionable_advice,
        educational_note=educational_note,
        metadata={"test": True},
    )


# --- 1. ReviewResult & ReviewDecision Domain Model Tests ---


def test_review_result_and_decision_schema_validation() -> None:
    """Test ReviewResult and ReviewDecision creation, validation, and JSON serialization."""
    eval_result = ReviewResult(
        evaluation_id="eval_01",
        response_id="resp_01",
        quality_score=8.5,
        confidence=0.95,
        reviewer_name="rule_based_v1",
        reviewer_type="rule_based",
        passed_checks=["coaching_quality", "safety"],
        failed_checks=[],
        reason_codes=[ReasonCode.GOOD_RISK_ANALYSIS, ReasonCode.GOOD_ACTION_PLAN],
        revision_suggestions=[],
        metadata={"raw": True},
    )

    assert eval_result.evaluation_id == "eval_01"
    assert eval_result.quality_score == 8.5

    engine = DecisionEngine(approval_threshold=7.0)
    decision = engine.evaluate_result(eval_result)

    assert decision.response_id == "resp_01"
    assert decision.verdict == ReviewVerdict.APPROVE
    assert decision.quality_score == 8.5
    assert len(decision.reason_codes) == 2

    # Test serialization
    dumped = decision.model_dump(mode="json")
    assert dumped["verdict"] == "APPROVE"
    assert dumped["reason_codes"] == ["GOOD_RISK_ANALYSIS", "GOOD_ACTION_PLAN"]


# --- 2. Quality Scorer & Criteria Suite Tests ---


def test_quality_scorer_calculations() -> None:
    """Test QualityScorer weighted quality calculation."""
    scorer = QualityScorer()
    breakdown = scorer.compute_score(
        reasoning_quality=8.0,
        coaching_usefulness=8.0,
        educational_value=8.0,
        consistency=8.0,
        completeness=8.0,
    )

    assert breakdown.overall_quality_score == 8.0
    assert breakdown.reasoning_quality == 8.0


def test_review_criteria_suite_default() -> None:
    """Test ReviewCriteriaSuite default instantiation and renamed criterion properties."""
    suite = ReviewCriteriaSuite.default_suite()
    assert len(suite.criteria) >= 10
    assert "coaching_quality" in suite.criteria
    assert "factual_consistency" in suite.criteria
    assert "safety" in suite.criteria
    assert suite.criteria["factual_consistency"].threshold == 7.0


# --- 3. RuleBasedReviewer & DecisionEngine Tests ---


def test_rule_based_reviewer_produces_raw_result() -> None:
    """Test RuleBasedReviewer produces raw ReviewResult without making policy decision."""
    reviewer = RuleBasedReviewer()
    resp = _create_sample_coach_response()

    eval_result = reviewer.review(resp)

    assert isinstance(eval_result, ReviewResult)
    assert eval_result.quality_score >= 7.0
    assert "factual_consistency" in eval_result.passed_checks
    assert len(eval_result.failed_checks) == 0


def test_decision_engine_verdict_thresholds() -> None:
    """Test DecisionEngine applies policy thresholds to render APPROVE, NEEDS_REVISION, REJECT."""
    engine = DecisionEngine(approval_threshold=7.0, revision_threshold=4.0)

    # Approving result
    res_good = ReviewResult(
        evaluation_id="e1",
        response_id="r1",
        quality_score=8.0,
        reviewer_name="rev1",
        reviewer_type="rule_based",
        passed_checks=["all"],
        failed_checks=[],
    )
    dec_good = engine.evaluate_result(res_good)
    assert dec_good.verdict == ReviewVerdict.APPROVE

    # Revision result
    res_mid = ReviewResult(
        evaluation_id="e2",
        response_id="r2",
        quality_score=5.5,
        reviewer_name="rev1",
        reviewer_type="rule_based",
        passed_checks=["some"],
        failed_checks=["actionability", "internal_consistency", "style"],
    )
    dec_mid = engine.evaluate_result(res_mid)
    assert dec_mid.verdict == ReviewVerdict.NEEDS_REVISION
    assert len(dec_mid.revision_suggestions) > 0

    # Reject result
    res_bad = ReviewResult(
        evaluation_id="e3",
        response_id="r3",
        quality_score=2.0,
        reviewer_name="rev1",
        reviewer_type="rule_based",
        passed_checks=[],
        failed_checks=["all"],
    )
    dec_bad = engine.evaluate_result(res_bad)
    assert dec_bad.verdict == ReviewVerdict.REJECT


def test_rule_based_reviewer_safety_rejection() -> None:
    """Test DecisionEngine rejects response when safety check fails."""
    reviewer = RuleBasedReviewer()
    pipeline = ReviewPipeline(reviewer=reviewer)
    resp = _create_sample_coach_response(
        educational_note="This setup guarantees 100% profit with zero risk."
    )

    decision = pipeline.run(resp)

    assert decision.verdict == ReviewVerdict.REJECT
    assert "safety" in decision.failed_checks
    assert ReasonCode.UNSAFE_CONTENT in decision.reason_codes


# --- 4. Review Strategies & Pipeline Tests ---


def test_single_reviewer_strategy_execution() -> None:
    """Test SingleReviewerStrategy runs specified reviewer returning ReviewResult."""
    strategy = SingleReviewerStrategy()
    reviewer = RuleBasedReviewer()
    resp = _create_sample_coach_response()
    suite = ReviewCriteriaSuite.default_suite()

    eval_result = strategy.execute(resp, reviewers=[reviewer], criteria_suite=suite)
    assert isinstance(eval_result, ReviewResult)
    assert eval_result.metadata["review_strategy"] == "single"


def test_review_pipeline_orchestration() -> None:
    """Test ReviewPipeline orchestrates evaluation and decision engine to record complete metadata."""
    pipeline = ReviewPipeline(reviewer=RuleBasedReviewer())
    resp = _create_sample_coach_response()

    decision = pipeline.run(resp)

    assert decision.verdict == ReviewVerdict.APPROVE
    assert "pipeline_name" in decision.metadata
    assert decision.metadata["pipeline_name"] == "review_pipeline"
    assert "criteria_version" in decision.metadata
    assert "configuration_version" in decision.metadata
    assert "policy_approval_threshold" in decision.metadata


def test_ai_mock_reviewers() -> None:
    """Test mock AI and human reviewers work as interchangeable extensions returning ReviewResult."""
    resp = _create_sample_coach_response()

    for rev in [
        GPTReviewer(),
        ClaudeReviewer(),
        GeminiReviewer(),
        ConsensusReviewer(),
        HumanReviewer(),
    ]:
        pipeline = ReviewPipeline(reviewer=rev)
        decision = pipeline.run(resp)
        assert isinstance(decision, ReviewDecision)
        assert decision.reviewer_name == rev.reviewer_name


def test_strategy_extension_scaffolds() -> None:
    """Test extension scaffolds raise NotImplementedError as designed for future milestones."""
    resp = _create_sample_coach_response()
    suite = ReviewCriteriaSuite.default_suite()

    for strat in [MultiReviewerStrategy(), ConsensusStrategy(), DebateStrategy()]:
        try:
            strat.execute(resp, reviewers=[RuleBasedReviewer()], criteria_suite=suite)
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass


# --- 5. Reason Codes & Revision Suggestions Tests ---


def test_revision_suggestion_generator() -> None:
    """Test RevisionSuggestionGenerator maps reason codes to actionable suggestions."""
    suggestions = RevisionSuggestionGenerator.generate_suggestions(
        failed_checks=["actionability", "internal_consistency"],
        reason_codes=[
            ReasonCode.MISSING_ACTIONABLE_ADVICE,
            ReasonCode.INCONSISTENT_REASONING,
        ],
        verdict=ReviewVerdict.NEEDS_REVISION,
    )

    assert len(suggestions) == 2
    assert "actionable advice" in suggestions[0] or "actionable advice" in suggestions[1]


# --- 6. Hydra Configuration Tests ---


def test_hydra_review_config_loading() -> None:
    """Test loading Hydra configuration includes review settings."""
    app_settings = load_hydra_config()
    assert hasattr(app_settings, "review")
    assert app_settings.review.strategy == "single"
    assert app_settings.review.reviewer == "rule_based"
    assert app_settings.review.approval_threshold == 7.0


# --- 7. CLI Subcommands Tests ---


def test_cli_review_run_sample() -> None:
    """Test `tsml review run` without args executes sample review."""
    result = runner.invoke(cli_app, ["review", "run"])
    assert result.exit_code == 0
    assert "Review Decision" in result.output


def test_cli_review_run_json(tmp_path: Path) -> None:
    """Test `tsml review run` with --json outputs valid JSON."""
    resp = _create_sample_coach_response()
    input_file = tmp_path / "coach_resp.json"
    input_file.write_text(resp.model_dump_json(), encoding="utf-8")

    result = runner.invoke(cli_app, ["review", "run", "-i", str(input_file), "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["response_id"] == resp.response_id
    assert parsed["verdict"] == "APPROVE"


def test_cli_review_batch_and_report(tmp_path: Path) -> None:
    """Test `tsml review batch` and `tsml review report` commands."""
    resp1 = _create_sample_coach_response("resp_01")
    resp2 = _create_sample_coach_response("resp_02", headline="Short", overall_score=2.0)
    batch_file = tmp_path / "batch_input.json"
    out_file = tmp_path / "batch_results.json"

    batch_file.write_text(
        json.dumps([resp1.model_dump(mode="json"), resp2.model_dump(mode="json")]),
        encoding="utf-8",
    )

    # Execute batch
    result_batch = runner.invoke(
        cli_app,
        ["review", "batch", "-i", str(batch_file), "-o", str(out_file)],
    )
    assert result_batch.exit_code == 0
    assert out_file.exists()

    # Execute report
    result_report = runner.invoke(cli_app, ["review", "report", "-i", str(out_file)])
    assert result_report.exit_code == 0
    assert "Review Pipeline Quality Summary" in result_report.output
    assert "Total Evaluated" in result_report.output
