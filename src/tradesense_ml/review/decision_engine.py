"""Centralized DecisionEngine converting raw ReviewResults into policy-driven ReviewDecisions."""

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from tradesense_ml.domain.schemas.review import (
    ReasonCode,
    ReviewDecision,
    ReviewResult,
    ReviewVerdict,
)
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.review.codes import RevisionSuggestionGenerator

logger = get_logger()


class DecisionEngine:
    """Policy decision engine converting raw ReviewResults into standard ReviewDecisions based on configurable thresholds."""

    def __init__(
        self,
        approval_threshold: float = 7.0,
        revision_threshold: float = 4.0,
    ) -> None:
        self.approval_threshold = approval_threshold
        self.revision_threshold = revision_threshold

    def evaluate_result(
        self,
        eval_result: ReviewResult,
        approval_threshold: float | None = None,
        revision_threshold: float | None = None,
        **kwargs: Any,
    ) -> ReviewDecision:
        """Apply decision policy rules to raw ReviewResult and render final ReviewDecision.

        Args:
            eval_result: Raw ReviewResult produced by a reviewer.
            approval_threshold: Optional threshold override for APPROVE verdict.
            revision_threshold: Optional threshold override for REJECT verdict.
            **kwargs: Additional metadata parameters.

        Returns:
            Structured ReviewDecision domain model.
        """
        start_time = time.perf_counter()
        app_thresh = (
            approval_threshold if approval_threshold is not None else self.approval_threshold
        )
        rev_thresh = (
            revision_threshold if revision_threshold is not None else self.revision_threshold
        )

        score = eval_result.quality_score
        failed_checks = list(eval_result.failed_checks)
        reason_codes = list(eval_result.reason_codes)

        # Policy decision rules
        if (
            ReasonCode.UNSAFE_CONTENT in reason_codes
            or "safety" in failed_checks
            or score < rev_thresh
        ):
            verdict = ReviewVerdict.REJECT
        elif score >= app_thresh and len(failed_checks) <= 2:
            verdict = ReviewVerdict.APPROVE
        else:
            verdict = ReviewVerdict.NEEDS_REVISION

        if verdict == ReviewVerdict.APPROVE and not reason_codes:
            reason_codes.append(ReasonCode.EXCELLENT_COACHING)

        # Generate structured revision suggestions
        revision_suggestions = RevisionSuggestionGenerator.generate_suggestions(
            failed_checks=failed_checks,
            reason_codes=reason_codes,
            verdict=verdict,
        )

        decision_duration_ms = (time.perf_counter() - start_time) * 1000.0

        decision_metadata = {
            **eval_result.metadata,
            "policy_approval_threshold": app_thresh,
            "policy_revision_threshold": rev_thresh,
            "decision_engine_duration_ms": round(decision_duration_ms, 2),
        }

        return ReviewDecision(
            review_id=f"rev_{uuid.uuid4().hex[:10]}",
            response_id=eval_result.response_id,
            verdict=verdict,
            quality_score=score,
            confidence=eval_result.confidence,
            reviewer_name=eval_result.reviewer_name,
            reviewer_type=eval_result.reviewer_type,
            review_timestamp=datetime.now(UTC),
            review_duration_ms=round(eval_result.metadata.get("reviewer_duration_ms", 0.0), 2),
            passed_checks=eval_result.passed_checks,
            failed_checks=failed_checks,
            reason_codes=reason_codes,
            revision_suggestions=revision_suggestions,
            metadata=decision_metadata,
        )
