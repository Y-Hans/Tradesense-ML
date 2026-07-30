"""Unit tests for review pipeline execution flow and audit trail."""

from datetime import datetime

from tradesense_ml.domain.schemas import (
    CoachRequest,
    CoachResponse,
    DisciplineEvaluation,
    ReviewAuditRecord,
    ReviewDecision,
    ReviewedExample,
    ReviewStage,
    RiskEvaluation,
    Side,
    Trade,
)
from tradesense_ml.pipelines.review.pipeline import BaseReviewStage, ReviewPipeline


class MockValidationStage(BaseReviewStage):
    """Mock validation stage."""

    def __init__(self) -> None:
        super().__init__(stage_enum=ReviewStage.AUTOMATED_VALIDATION)

    def process(self, example: ReviewedExample) -> tuple[ReviewedExample, ReviewAuditRecord]:
        record = ReviewAuditRecord(
            record_id="rec_val_1",
            stage=self.stage_enum,
            reviewer_id="automated_validator",
            decision=ReviewDecision.APPROVE,
            score=10.0,
            comments="Schema and rules validated.",
        )
        return example, record


class MockAITeacherReviewStage(BaseReviewStage):
    """Mock AI teacher review stage."""

    def __init__(self) -> None:
        super().__init__(stage_enum=ReviewStage.AI_TEACHER_REVIEW)

    def process(self, example: ReviewedExample) -> tuple[ReviewedExample, ReviewAuditRecord]:
        record = ReviewAuditRecord(
            record_id="rec_ai_1",
            stage=self.stage_enum,
            reviewer_id="teacher_claude35",
            decision=ReviewDecision.APPROVE,
            score=9.5,
            comments="Consensus review approved.",
        )
        return example, record


def test_review_pipeline_flow() -> None:
    """Test 4-stage review pipeline execution and audit trail accumulation."""
    trade = Trade(
        trade_id="t1",
        user_id="u1",
        symbol="BTC/USD",
        side=Side.BUY,
        entry_price=50000.0,
        quantity=1.0,
        entry_timestamp=datetime.utcnow(),
    )
    req = CoachRequest(request_id="r1", user_id="u1", trade=trade)
    risk = RiskEvaluation(
        risk_score=8.0,
        position_size_compliant=True,
        stop_loss_defined=True,
        risk_summary="Good risk",
    )
    disc = DisciplineEvaluation(
        discipline_score=9.0,
        fomo_indicator=False,
        revenge_trade_indicator=False,
        overtrading_indicator=False,
        plan_adherence_score=9.0,
        discipline_summary="Disciplined trade",
    )
    resp = CoachResponse(
        response_id="res1",
        request_id="r1",
        headline="Great trade",
        overall_score=8.5,
        risk_evaluation=risk,
        discipline_evaluation=disc,
        actionable_advice=["Maintain stop loss discipline"],
        educational_note="Risk reward ratio was well managed.",
    )

    example = ReviewedExample(
        example_id="ex_1",
        request=req,
        teacher_response=resp,
        review_status=ReviewStage.AUTOMATED_VALIDATION,
    )

    pipeline = ReviewPipeline([MockValidationStage(), MockAITeacherReviewStage()])
    reviewed = pipeline.run(example)

    assert reviewed.review_status == ReviewStage.APPROVED
    assert len(reviewed.audit_trail) == 2
    assert reviewed.audit_trail[0].reviewer_id == "automated_validator"
    assert reviewed.audit_trail[1].reviewer_id == "teacher_claude35"
