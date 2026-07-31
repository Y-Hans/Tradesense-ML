"""Configurable dataset example filtering engine."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.coaching import CoachResponse
from tradesense_ml.domain.schemas.examples import ReviewedExample
from tradesense_ml.domain.schemas.review import ReviewDecision, ReviewStage, ReviewVerdict
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class DatasetFilterConfig(BaseModel):
    """Configuration options for dataset filtering."""

    model_config = ConfigDict(frozen=True)

    only_approved: bool = Field(
        default=True, description="Keep only examples with approved review verdicts"
    )
    min_quality_score: float = Field(
        default=7.0, ge=0.0, le=10.0, description="Minimum allowed quality score"
    )
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum allowed reviewer confidence"
    )
    allowed_review_versions: list[str] | None = Field(
        default=None, description="Whitelist of allowed review versions"
    )
    allowed_teacher_versions: list[str] | None = Field(
        default=None, description="Whitelist of allowed teacher model names/versions"
    )
    remove_duplicates: bool = Field(
        default=True, description="Deduplicate items based on response_id or request_id"
    )
    remove_incomplete: bool = Field(
        default=True, description="Remove items with missing required fields or empty content"
    )


class FilteringResult(BaseModel):
    """Result payload from filtering operation."""

    model_config = ConfigDict(frozen=True)

    kept_examples: list[Any] = Field(..., description="List of accepted example records")
    total_evaluated: int = Field(..., description="Total input items evaluated")
    rejected_count: int = Field(..., description="Total rejected items")
    duplicate_count: int = Field(..., description="Deduplicated items count")
    incomplete_count: int = Field(..., description="Incomplete items count")
    rejection_reasons: dict[str, int] = Field(
        default_factory=dict, description="Breakdown of rejection reason counts"
    )


class DatasetFilter:
    """Configurable filter for reviewed dataset examples."""

    def __init__(self, config: DatasetFilterConfig | None = None) -> None:
        self.config = config or DatasetFilterConfig()

    def filter_batch(self, items: list[Any]) -> FilteringResult:
        """Filter a batch of reviewed examples or request-response-decision tuples."""
        kept: list[Any] = []
        seen_ids: set[str] = set()
        rejected_count = 0
        duplicate_count = 0
        incomplete_count = 0
        rejection_reasons: dict[str, int] = {}

        for item in items:
            is_valid, reason = self._eval_item(item)

            if not is_valid:
                rejected_count += 1
                if reason:
                    if "duplicate" in reason:
                        duplicate_count += 1
                    if "incomplete" in reason:
                        incomplete_count += 1
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue

            # Deduplication check
            item_id = self._extract_id(item)
            if self.config.remove_duplicates and item_id in seen_ids:
                duplicate_count += 1
                rejected_count += 1
                rejection_reasons["duplicate_id"] = rejection_reasons.get("duplicate_id", 0) + 1
                continue

            seen_ids.add(item_id)
            kept.append(item)

        logger.info(
            f"Filtered dataset batch: {len(items)} evaluated -> {len(kept)} kept, {rejected_count} rejected ({duplicate_count} duplicates)"
        )

        return FilteringResult(
            kept_examples=kept,
            total_evaluated=len(items),
            rejected_count=rejected_count,
            duplicate_count=duplicate_count,
            incomplete_count=incomplete_count,
            rejection_reasons=rejection_reasons,
        )

    def _eval_item(self, item: Any) -> tuple[bool, str | None]:
        """Evaluate a single item against filter criteria."""
        resp: CoachResponse | None = None
        decision: ReviewDecision | None = None
        status: str | None = None
        score: float | None = None
        confidence: float | None = None

        if isinstance(item, ReviewedExample):
            resp = item.teacher_response
            status = (
                item.review_status.value
                if hasattr(item.review_status, "value")
                else str(item.review_status)
            )
            score = item.final_quality_score
        elif isinstance(item, dict):
            resp = (
                CoachResponse.model_validate(item["teacher_response"])
                if "teacher_response" in item
                else None
            )
            if "review_decision" in item:
                decision = ReviewDecision.model_validate(item["review_decision"])
                status = decision.verdict.value
                score = decision.quality_score
                confidence = decision.confidence
            elif "review_status" in item:
                status = str(item["review_status"])
                score = item.get("final_quality_score")
        elif isinstance(item, tuple) and len(item) >= 3:
            _, resp, decision = item[0], item[1], item[2]
            if isinstance(decision, ReviewDecision):
                status = decision.verdict.value
                score = decision.quality_score
                confidence = decision.confidence

        # Incomplete check
        if self.config.remove_incomplete:
            if resp is None:
                return False, "incomplete_missing_response"
            if not resp.headline or not resp.actionable_advice or not resp.educational_note:
                return False, "incomplete_empty_fields"

        # Verdict check
        if self.config.only_approved:
            approved_states = [
                ReviewVerdict.APPROVE.value,
                ReviewStage.APPROVED.value,
                "APPROVE",
                "APPROVED",
            ]
            if status is not None and status not in approved_states:
                return False, f"unapproved_verdict_{status}"

        # Quality score check
        if score is not None and score < self.config.min_quality_score:
            return False, f"below_min_quality_score_{score}"

        # Confidence check
        if confidence is not None and confidence < self.config.min_confidence:
            return False, f"below_min_confidence_{confidence}"

        # Teacher model whitelist check
        if self.config.allowed_teacher_versions and resp:
            teacher_model = resp.metadata.get("model") or resp.metadata.get("provider")
            if teacher_model and teacher_model not in self.config.allowed_teacher_versions:
                return False, f"disallowed_teacher_{teacher_model}"

        return True, None

    def _extract_id(self, item: Any) -> str:
        """Extract unique identifier from item."""
        if hasattr(item, "example_id"):
            return str(item.example_id)
        if isinstance(item, dict) and "example_id" in item:
            return str(item["example_id"])
        if isinstance(item, dict) and "teacher_response" in item:
            return str(item["teacher_response"].get("response_id", id(item)))
        if isinstance(item, tuple) and len(item) >= 2 and hasattr(item[1], "response_id"):
            return str(item[1].response_id)
        return str(id(item))
