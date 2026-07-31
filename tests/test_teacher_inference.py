"""Comprehensive test suite for Teacher Inference Pipeline, strategies, parser, validator, retry, and CLI."""

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tradesense_ml.cli.main import app
from tradesense_ml.domain.schemas.coaching import CoachRequest, CoachResponse
from tradesense_ml.domain.schemas.market_context import MarketContext, MarketRegime, VolatilityLevel
from tradesense_ml.domain.schemas.teacher import (
    RenderedPrompt,
    TeacherRequest,
)
from tradesense_ml.domain.schemas.trade import Side, Trade
from tradesense_ml.pipelines.inference.pipeline import TeacherInferencePipeline
from tradesense_ml.pipelines.inference.strategies import SingleTeacherStrategy
from tradesense_ml.teachers.base import BaseTeacherProvider
from tradesense_ml.teachers.prompt_builder import PromptBuilder
from tradesense_ml.teachers.prompt_renderer import PromptRenderer
from tradesense_ml.teachers.response_parser import ResponseParser, ResponseParsingError
from tradesense_ml.teachers.retry import RetryConfig, RetryHandler
from tradesense_ml.teachers.router import TeacherRouter
from tradesense_ml.teachers.validator import ResponseValidationError, ResponseValidator

runner = CliRunner()


@pytest.fixture
def sample_trade() -> Trade:
    """Fixture providing a sample Trade model."""
    return Trade(
        trade_id="t_test_100",
        user_id="u_trader_1",
        symbol="BTC/USD",
        side=Side.BUY,
        entry_price=65000.0,
        exit_price=64000.0,
        quantity=1.5,
        initial_stop_loss=64500.0,
        initial_take_profit=68000.0,
        pnl=-1500.0,
        entry_timestamp="2026-07-30T10:00:00Z",
        exit_timestamp="2026-07-30T11:00:00Z",
    )


@pytest.fixture
def sample_market_context() -> MarketContext:
    """Fixture providing a sample MarketContext model."""
    return MarketContext(
        context_id="ctx_test_01",
        symbol="BTC/USD",
        regime=MarketRegime.BULLISH_TREND,
        volatility=VolatilityLevel.HIGH,
    )


@pytest.fixture
def sample_coach_request(sample_trade: Trade, sample_market_context: MarketContext) -> CoachRequest:
    """Fixture providing a sample CoachRequest model."""
    return CoachRequest(
        request_id="req_test_99",
        user_id="u_trader_1",
        trade=sample_trade,
        market_context=sample_market_context,
        user_notes="Attempted trend continuation entry.",
        requested_aspects=["risk", "discipline"],
    )


@pytest.fixture
def mock_valid_response_dict(sample_coach_request: CoachRequest) -> dict:
    """Fixture returning a valid CoachResponse data dictionary."""
    return {
        "response_id": f"resp_{sample_coach_request.request_id}",
        "request_id": sample_coach_request.request_id,
        "headline": "Disciplined trade with sub-optimal stop loss execution.",
        "overall_score": 7.5,
        "risk_evaluation": {
            "risk_score": 7.0,
            "risk_reward_ratio": 2.0,
            "position_size_compliant": True,
            "stop_loss_defined": True,
            "max_drawdown_risk_pct": 2.5,
            "risk_summary": "Risk parameters adhered to guidelines.",
            "reason_codes": [],
        },
        "discipline_evaluation": {
            "discipline_score": 8.0,
            "fomo_indicator": False,
            "revenge_trade_indicator": False,
            "overtrading_indicator": False,
            "plan_adherence_score": 8.0,
            "discipline_summary": "No FOMO or revenge trading observed.",
            "reason_codes": [],
        },
        "actionable_advice": [
            "Set stop loss closer to technical support.",
            "Reduce leverage during high volatility.",
        ],
        "educational_note": "Risk/Reward ratio management is vital during volatile trends.",
        "metadata": {},
    }


class CustomMockTeacherProvider(BaseTeacherProvider):
    """Custom mock provider returning configurable JSON responses."""

    def __init__(
        self, provider_name: str = "custom_mock", response_payload: str | None = None
    ) -> None:
        super().__init__(
            provider_name=provider_name,
            default_model="mock-v1",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
        )
        self.response_payload = response_payload

    def _do_generate(self, request: TeacherRequest) -> tuple[str, dict | None, int, int]:
        if self.response_payload is not None:
            return self.response_payload, None, 100, 200
        mock_data = {
            "response_id": f"resp_{request.request_id}",
            "request_id": request.request_id,
            "headline": "Solid execution with minor risk adjustments needed.",
            "overall_score": 8.0,
            "risk_evaluation": {
                "risk_score": 8.0,
                "position_size_compliant": True,
                "stop_loss_defined": True,
                "risk_summary": "Proper risk management.",
                "reason_codes": [],
            },
            "discipline_evaluation": {
                "discipline_score": 8.0,
                "fomo_indicator": False,
                "revenge_trade_indicator": False,
                "overtrading_indicator": False,
                "plan_adherence_score": 8.0,
                "discipline_summary": "Followed trading plan.",
                "reason_codes": [],
            },
            "actionable_advice": ["Maintain strict risk limits."],
            "educational_note": "Consistency is key.",
        }
        return json.dumps(mock_data), mock_data, 120, 250


def test_prompt_builder(sample_coach_request: CoachRequest) -> None:
    """Test PromptBuilder context extraction."""
    builder = PromptBuilder()
    context = builder.build_context(sample_coach_request, prompt_version="v1")

    assert context.request_id == sample_coach_request.request_id
    assert "BTC/USD" in context.trade_record_text
    assert "BULLISH_TREND" in context.market_context_text
    assert "Attempted trend continuation entry." in context.user_notes_text
    assert "risk, discipline" in context.requested_aspects_text


def test_prompt_renderer(sample_coach_request: CoachRequest) -> None:
    """Test PromptRenderer producing provider-independent RenderedPrompt."""
    builder = PromptBuilder()
    context = builder.build_context(sample_coach_request, prompt_version="v1")
    renderer = PromptRenderer()

    rendered = renderer.render(context)
    assert isinstance(rendered, RenderedPrompt)
    assert "TradeSense Coach" in rendered.system_prompt
    assert "BTC/USD" in rendered.user_prompt
    assert rendered.prompt_version == "v1"

    teacher_req = rendered.to_teacher_request(request_id="req_test_99")
    assert isinstance(teacher_req, TeacherRequest)
    assert teacher_req.system_prompt == rendered.system_prompt


def test_response_parser_markdown_and_raw_json(mock_valid_response_dict: dict) -> None:
    """Test ResponseParser handling raw json dict, string, and markdown-wrapped JSON."""
    raw_str = json.dumps(mock_valid_response_dict)
    parsed = ResponseParser.parse(raw_str, request_id="req_test_99")
    assert parsed.headline == mock_valid_response_dict["headline"]

    # Markdown wrapped
    markdown_str = f"```json\n{raw_str}\n```"
    parsed_md = ResponseParser.parse(markdown_str, request_id="req_test_99")
    assert parsed_md.overall_score == 7.5

    # Invalid JSON string
    with pytest.raises(ResponseParsingError):
        ResponseParser.parse("This is invalid json text", request_id="req_test_99")


def test_response_validator(mock_valid_response_dict: dict) -> None:
    """Test ResponseValidator checking score boundaries and non-empty summaries."""
    import copy

    valid_resp = CoachResponse.model_validate(mock_valid_response_dict)
    res = ResponseValidator.validate(valid_resp)
    assert res.is_valid
    assert not res.errors

    # Invalid empty headline / summary
    invalid_dict = copy.deepcopy(mock_valid_response_dict)
    invalid_dict["headline"] = ""
    invalid_resp = CoachResponse.model_validate(invalid_dict)
    res_inv = ResponseValidator.validate(invalid_resp)
    assert not res_inv.is_valid
    assert any("Headline is missing" in err for err in res_inv.errors)

    with pytest.raises(ResponseValidationError):
        ResponseValidator.validate_and_raise(invalid_resp)


def test_retry_handler() -> None:
    """Test RetryHandler backoff and retry behavior."""
    config = RetryConfig(max_retries=2, initial_backoff_sec=0.01, backoff_factor=1.0)
    handler = RetryHandler(config)

    attempts = 0

    def failing_func() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ResponseParsingError("Temporary JSON error", raw_content="")
        return "success"

    result = handler.execute(failing_func)
    assert result == "success"
    assert attempts == 2


def test_teacher_inference_pipeline_end_to_end(sample_coach_request: CoachRequest) -> None:
    """Test full TeacherInferencePipeline execution with mock provider and cost tracking."""
    mock_prov = CustomMockTeacherProvider()
    router = TeacherRouter([mock_prov])

    pipeline = TeacherInferencePipeline(router=router, strategy=SingleTeacherStrategy())
    response = pipeline.run(sample_coach_request, provider="custom_mock")

    assert isinstance(response, CoachResponse)
    assert response.request_id == sample_coach_request.request_id
    assert response.overall_score == 8.0
    assert response.metadata["provider"] == "custom_mock"
    assert response.metadata["token_usage"]["total_tokens"] > 0
    assert response.metadata["token_usage"]["estimated_cost_usd"] > 0.0


def test_cli_commands(sample_coach_request: CoachRequest) -> None:
    """Test CLI commands: tsml teacher providers, infer, batch."""
    # 1. Test providers
    res_prov = runner.invoke(app, ["teacher", "providers"])
    assert res_prov.exit_code == 0
    assert "openrouter" in res_prov.stdout.lower()

    # 2. Test single infer
    with tempfile.TemporaryDirectory() as tmp_dir:
        req_file = Path(tmp_dir) / "request.json"
        req_file.write_text(sample_coach_request.model_dump_json(), encoding="utf-8")
        out_file = Path(tmp_dir) / "output.json"

        res_infer = runner.invoke(
            app,
            [
                "teacher",
                "infer",
                "-f",
                str(req_file),
                "-p",
                "openrouter",
                "-o",
                str(out_file),
            ],
        )
        assert res_infer.exit_code == 0
        assert out_file.exists()

        # 3. Test batch
        batch_out_dir = Path(tmp_dir) / "batch_out"
        res_batch = runner.invoke(
            app,
            [
                "teacher",
                "batch",
                "-i",
                str(req_file),
                "-o",
                str(batch_out_dir),
                "-p",
                "openrouter",
            ],
        )
        assert res_batch.exit_code == 0
        assert len(list(batch_out_dir.glob("*.json"))) == 1
