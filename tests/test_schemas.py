"""Unit tests for Pydantic domain schemas and dataset lineage serialization."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from tradesense_ml.domain.schemas import (
    CoachRequest,
    CoachResponse,
    DatasetSplit,
    DatasetVersionMetadata,
    DisciplineEvaluation,
    MarketContext,
    MarketRegime,
    ReasonCodeCategory,
    ReasonCodeDetail,
    ReviewStage,
    RiskEvaluation,
    Side,
    StandardReasonCode,
    Trade,
    VolatilityLevel,
)


def test_trade_schema_valid() -> None:
    """Test valid Trade model instantiation and JSON serialization."""
    trade = Trade(
        trade_id="t100",
        user_id="u42",
        symbol="AAPL",
        side=Side.BUY,
        entry_price=150.0,
        exit_price=155.0,
        quantity=100.0,
        pnl=500.0,
        pnl_percentage=3.33,
        entry_timestamp=datetime.utcnow(),
    )
    assert trade.trade_id == "t100"
    json_str = trade.model_dump_json()
    reconstructed = Trade.model_validate_json(json_str)
    assert reconstructed.trade_id == trade.trade_id
    assert reconstructed.side == Side.BUY


def test_trade_schema_invalid_price() -> None:
    """Test invalid entry_price raises validation error."""
    with pytest.raises(ValidationError):
        Trade(
            trade_id="t101",
            user_id="u42",
            symbol="AAPL",
            side=Side.BUY,
            entry_price=-10.0,  # Invalid price <= 0
            quantity=10.0,
            entry_timestamp=datetime.utcnow(),
        )


def test_market_context_schema() -> None:
    """Test MarketContext schema instantiation."""
    ctx = MarketContext(
        context_id="mc1",
        symbol="BTC/USD",
        regime=MarketRegime.BULLISH_TREND,
        volatility=VolatilityLevel.HIGH,
        support_levels=[60000.0, 58000.0],
        resistance_levels=[65000.0, 70000.0],
    )
    assert ctx.regime == MarketRegime.BULLISH_TREND
    assert len(ctx.support_levels) == 2


def test_coaching_and_evaluation_schemas() -> None:
    """Test Risk, Discipline, CoachRequest, and CoachResponse schemas."""
    reason = ReasonCodeDetail(
        code=StandardReasonCode.D_CHASE_ENTRY_FOMO,
        category=ReasonCodeCategory.DISCIPLINE,
        explanation="Entered near local high without waiting for pull back.",
    )

    risk = RiskEvaluation(
        risk_score=7.5,
        risk_reward_ratio=2.0,
        position_size_compliant=True,
        stop_loss_defined=True,
        risk_summary="Acceptable risk parameters",
        reason_codes=[],
    )

    discipline = DisciplineEvaluation(
        discipline_score=4.0,
        fomo_indicator=True,
        revenge_trade_indicator=False,
        overtrading_indicator=False,
        plan_adherence_score=5.0,
        discipline_summary="FOMO entry detected",
        reason_codes=[reason],
    )

    trade = Trade(
        trade_id="t1",
        user_id="u1",
        symbol="ETH/USD",
        side=Side.LONG,
        entry_price=3000.0,
        quantity=1.0,
        entry_timestamp=datetime.utcnow(),
    )

    req = CoachRequest(request_id="req1", user_id="u1", trade=trade)
    resp = CoachResponse(
        response_id="resp1",
        request_id="req1",
        headline="Disciplined risk but chased entry",
        overall_score=5.75,
        risk_evaluation=risk,
        discipline_evaluation=discipline,
        actionable_advice=["Wait for 5m EMA pullback before entry"],
        educational_note="Chasing rallies increases slippage and drawdown probability.",
    )

    assert req.request_id == "req1"
    assert resp.risk_evaluation.risk_score == 7.5

    assert resp.discipline_evaluation.reason_codes[0].code == StandardReasonCode.D_CHASE_ENTRY_FOMO


def test_dataset_version_metadata_lineage() -> None:
    """Test full dataset lineage metadata schema."""
    lineage = DatasetVersionMetadata(
        dataset_id="tradesense_coaching_v1",
        dataset_version="1.0.0",
        teacher_model="anthropic/claude-3.5-sonnet",
        prompt_version="v1",
        rubric_version="v1",
        generator_version="0.1.0",
        review_version="1.0.0",
        source_hash="abcd1234hash",
        review_status=ReviewStage.APPROVED,
        quality_score=9.2,
        split=DatasetSplit.TRAIN,
    )

    assert lineage.teacher_model == "anthropic/claude-3.5-sonnet"
    assert lineage.review_status == ReviewStage.APPROVED
