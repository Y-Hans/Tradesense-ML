"""Dataset transformation engine for converting reviewed coaching examples into canonical DatasetExample objects."""

from typing import Any

from tradesense_ml.domain.schemas.coaching import CoachRequest, CoachResponse
from tradesense_ml.domain.schemas.dataset import DatasetExample
from tradesense_ml.domain.schemas.examples import ReviewedExample
from tradesense_ml.domain.schemas.review import ReviewDecision
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class DatasetTransformer:
    """Transformer converting reviewed coaching records into canonical DatasetExample objects."""

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are TradeSense AI, an expert trading coach. Analyze the user's trade execution "
        "and market context, evaluate risk and discipline metrics, and deliver clear, actionable, "
        "and educational coaching guidance."
    )

    def __init__(self, system_instruction: str | None = None) -> None:
        self.system_instruction = system_instruction or self.DEFAULT_SYSTEM_INSTRUCTION

    def transform_batch(
        self,
        items: list[Any],
        target_format: str = "sft_instruction",
        version: str = "v1.0.0",
    ) -> list[DatasetExample]:
        """Transform a list of reviewed examples into canonical DatasetExample records."""
        transformed: list[DatasetExample] = []

        for item in items:
            example = self.transform_single(item, target_format=target_format, version=version)
            if example:
                transformed.append(example)

        logger.info(
            f"Transformed {len(items)} items into {len(transformed)} '{target_format}' DatasetExample records."
        )
        return transformed

    def transform_single(
        self,
        item: Any,
        target_format: str = "sft_instruction",
        version: str = "v1.0.0",
    ) -> DatasetExample | None:
        """Transform a single reviewed item into a canonical DatasetExample object."""
        req: CoachRequest | None = None
        resp: CoachResponse | None = None
        decision: ReviewDecision | None = None
        example_id: str | None = None
        quality_score: float = 0.0

        if isinstance(item, ReviewedExample):
            example_id = item.example_id
            req = item.request
            resp = item.teacher_response
            quality_score = item.final_quality_score or 0.0
        elif isinstance(item, tuple) and len(item) >= 3:
            req, resp, decision = item[0], item[1], item[2]
            example_id = f"ex_{resp.response_id}"
            if isinstance(decision, ReviewDecision):
                quality_score = decision.quality_score
        elif isinstance(item, dict):
            example_id = item.get("example_id", f"ex_{item.get('response_id', 'unknown')}")
            req = CoachRequest.model_validate(item["request"]) if "request" in item else None
            resp = (
                CoachResponse.model_validate(item["teacher_response"])
                if "teacher_response" in item
                else None
            )
            if "review_decision" in item and isinstance(item["review_decision"], dict):
                decision = ReviewDecision.model_validate(item["review_decision"])
                quality_score = decision.quality_score

        if not req or not resp:
            return None

        # Build formatted text components
        input_text = self._format_input_text(req)
        output_text = self._format_output_text(resp)
        prompt_text = f"{self.system_instruction}\n\n### User Request:\n{input_text}\n\n### Coaching Response:\n"

        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]

        review_info = {
            "quality_score": quality_score,
            "verdict": decision.verdict.value if decision else "APPROVED",
            "reviewer_name": decision.reviewer_name if decision else "system",
            "reason_codes": (
                [rc.value if hasattr(rc, "value") else str(rc) for rc in decision.reason_codes]
                if decision
                else []
            ),
        }

        lineage_info = {
            "request_id": req.request_id,
            "response_id": resp.response_id,
            "review_id": decision.review_id if decision else "N/A",
            "teacher_provider": resp.metadata.get("provider", "unknown"),
            "teacher_model": resp.metadata.get("model", "unknown"),
        }

        raw_example = DatasetExample(
            example_id=example_id or f"ex_{resp.response_id}",
            instruction=self.system_instruction,
            input=input_text,
            output=output_text,
            messages=messages,
            prompt=prompt_text,
            reasoning=resp.metadata.get("reasoning"),
            format_type="canonical",
            review_info=review_info,
            lineage=lineage_info,
            metadata={
                "dataset_version": version,
                "user_id": req.user_id,
                "symbol": req.trade.symbol,
            },
        )

        from tradesense_ml.dataset.formats.registry import DatasetFormatRegistry

        format_strategy = DatasetFormatRegistry.get_format(target_format)
        return format_strategy.format_example(raw_example)

    def _format_input_text(self, req: CoachRequest) -> str:
        """Format CoachRequest into user input prompt text."""
        trade = req.trade
        market = req.market_context

        lines = [
            "Trade Details:",
            f"- Symbol: {trade.symbol}",
            f"- Side: {trade.side.value}",
            f"- Entry Price: ${trade.entry_price:.2f}",
            f"- Quantity: {trade.quantity}",
        ]

        if trade.exit_price is not None:
            lines.append(f"- Exit Price: ${trade.exit_price:.2f}")
        if trade.pnl is not None:
            lines.append(f"- Realized PnL: ${trade.pnl:.2f} ({trade.pnl_percentage or 0.0:.2f}%)")

        if market:
            regime_str = (
                market.regime.value if hasattr(market.regime, "value") else str(market.regime)
            )
            vol_str = (
                market.volatility.value
                if hasattr(market.volatility, "value")
                else str(market.volatility)
            )
            lines.extend(
                [
                    "\nMarket Context:",
                    f"- Regime: {regime_str}",
                    f"- Volatility: {vol_str}",
                ]
            )
            trend_score = getattr(market, "overall_trend_score", None)
            if trend_score is not None:
                lines.append(f"- Overall Trend Score: {trend_score:.2f}")

        if req.user_notes:
            lines.append(f'\nTrader Notes: "{req.user_notes}"')

        return "\n".join(lines)

    def _format_output_text(self, resp: CoachResponse) -> str:
        """Format CoachResponse into target output coaching text."""
        risk = resp.risk_evaluation
        disc = resp.discipline_evaluation

        advice_formatted = "\n".join(f"- {item}" for item in resp.actionable_advice)

        return (
            f"## Coaching Summary\n{resp.headline}\n\n"
            f"### Risk Evaluation (Score: {risk.risk_score:.1f}/10.0)\n"
            f"{risk.risk_summary}\n\n"
            f"### Discipline Evaluation (Score: {disc.discipline_score:.1f}/10.0)\n"
            f"{disc.discipline_summary}\n\n"
            f"### Actionable Advice\n{advice_formatted}\n\n"
            f"### Educational Concept\n{resp.educational_note}"
        )
