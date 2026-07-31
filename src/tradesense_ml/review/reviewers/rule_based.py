"""RuleBasedReviewer implementation for deterministic quality evaluation."""

import time
import uuid
from typing import Any

from tradesense_ml.domain.schemas.coaching import CoachResponse
from tradesense_ml.domain.schemas.review import ReasonCode, ReviewResult
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.review.base import BaseReviewer
from tradesense_ml.review.criteria import ReviewCriteriaSuite
from tradesense_ml.review.scoring import QualityScorer

logger = get_logger()


class RuleBasedReviewer(BaseReviewer):
    """Deterministic, rule-based response reviewer executing heuristics across 10 criteria dimensions."""

    UNSAFE_PATTERNS = [
        "guaranteed 100% profit",
        "guarantees 100% profit",
        "100% profit",
        "risk-free infinite gain",
        "never use a stop loss",
        "all in 100x leverage without stop",
        "cannot lose money",
    ]

    GENERIC_HEADLINES = ["good trade", "bad trade", "trade review", "ok", "n/a"]

    def __init__(self, reviewer_name: str = "rule_based_v1") -> None:
        super().__init__(reviewer_name=reviewer_name, reviewer_type="rule_based")
        self.scorer = QualityScorer()

    def review(
        self,
        response: CoachResponse,
        criteria_suite: ReviewCriteriaSuite | None = None,
        **kwargs: Any,
    ) -> ReviewResult:
        """Evaluate CoachResponse against deterministic rules and criteria thresholds, returning raw ReviewResult."""
        start_time = time.perf_counter()
        suite = criteria_suite or ReviewCriteriaSuite.default_suite()

        passed_checks: list[str] = []
        failed_checks: list[str] = []
        reason_codes: list[ReasonCode] = []

        # 1. Coaching Quality
        coaching_score = 10.0
        headline = response.headline.strip().lower()
        if len(headline) < 10 or headline in self.GENERIC_HEADLINES:
            coaching_score -= 4.0
            failed_checks.append("coaching_quality")
            reason_codes.append(ReasonCode.STYLE_VIOLATION)
        else:
            passed_checks.append("coaching_quality")

        # 2. Risk Analysis Quality
        risk_score = 10.0
        if (
            not response.risk_evaluation.risk_summary
            or len(response.risk_evaluation.risk_summary.strip()) < 10
        ):
            risk_score -= 5.0
            failed_checks.append("risk_analysis_quality")
            reason_codes.append(ReasonCode.INSUFFICIENT_EXPLANATION)
        else:
            passed_checks.append("risk_analysis_quality")
            reason_codes.append(ReasonCode.GOOD_RISK_ANALYSIS)

        # 3. Discipline Analysis Quality
        discipline_score = 10.0
        if (
            not response.discipline_evaluation.discipline_summary
            or len(response.discipline_evaluation.discipline_summary.strip()) < 10
        ):
            discipline_score -= 5.0
            failed_checks.append("discipline_analysis_quality")
            reason_codes.append(ReasonCode.INSUFFICIENT_EXPLANATION)
        else:
            passed_checks.append("discipline_analysis_quality")
            reason_codes.append(ReasonCode.GOOD_DISCIPLINE_ANALYSIS)

        # 4. Internal Consistency
        consistency_score = 10.0
        expected_avg_score = (
            response.risk_evaluation.risk_score + response.discipline_evaluation.discipline_score
        ) / 2.0
        score_diff = abs(response.overall_score - expected_avg_score)
        if score_diff > 3.0:
            consistency_score -= min(score_diff * 2.0, 7.0)
            failed_checks.append("internal_consistency")
            reason_codes.append(ReasonCode.INCONSISTENT_REASONING)
        else:
            passed_checks.append("internal_consistency")

        # 5. Educational Value
        educational_score = 10.0
        if not response.educational_note or len(response.educational_note.strip()) < 15:
            educational_score -= 6.0
            failed_checks.append("educational_value")
            reason_codes.append(ReasonCode.LOW_EDUCATIONAL_VALUE)
        else:
            passed_checks.append("educational_value")

        # 6. Actionability
        actionability_score = 10.0
        if not response.actionable_advice or len(response.actionable_advice) == 0:
            actionability_score = 0.0
            failed_checks.append("actionability")
            reason_codes.append(ReasonCode.MISSING_ACTIONABLE_ADVICE)
        else:
            short_items = [item for item in response.actionable_advice if len(item.strip()) < 8]
            if short_items:
                actionability_score -= min(len(short_items) * 3.0, 6.0)
                failed_checks.append("actionability")
                reason_codes.append(ReasonCode.MISSING_ACTIONABLE_ADVICE)
            else:
                passed_checks.append("actionability")
                reason_codes.append(ReasonCode.GOOD_ACTION_PLAN)

        # 7. Completeness
        completeness_score = 10.0
        if not response.headline or not response.actionable_advice or not response.educational_note:
            completeness_score = 2.0
            failed_checks.append("completeness")
            reason_codes.append(ReasonCode.INCOMPLETE_RESPONSE)
        else:
            passed_checks.append("completeness")

        # 8. Factual Consistency (Renamed from hallucination_detection)
        factual_score = 10.0
        combined_text = f"{response.headline} {response.educational_note} {' '.join(response.actionable_advice)}".lower()
        if "guaranteed profit" in combined_text or "infinite gain" in combined_text:
            factual_score = 0.0
            failed_checks.append("factual_consistency")
            reason_codes.append(ReasonCode.HALLUCINATED_MARKET_FACT)
        else:
            passed_checks.append("factual_consistency")

        # 9. Style Consistency
        style_score = 10.0
        if len(response.headline) > 200:
            style_score -= 4.0
            failed_checks.append("style_consistency")
            reason_codes.append(ReasonCode.STYLE_VIOLATION)
        else:
            passed_checks.append("style_consistency")

        # 10. Safety
        safety_score = 10.0
        for pattern in self.UNSAFE_PATTERNS:
            if pattern in combined_text:
                safety_score = 0.0
                failed_checks.append("safety")
                reason_codes.append(ReasonCode.UNSAFE_CONTENT)
                break
        if safety_score == 10.0:
            passed_checks.append("safety")

        # Compute aggregate quality score using QualityScorer
        breakdown = self.scorer.compute_score(
            reasoning_quality=(risk_score + discipline_score) / 2.0,
            coaching_usefulness=(coaching_score + actionability_score) / 2.0,
            educational_value=educational_score,
            consistency=consistency_score,
            completeness=completeness_score,
            extra_metrics={"factual_consistency": factual_score},
        )

        # Deduplicate reason codes preserving order
        unique_reason_codes: list[ReasonCode] = []
        for rc in reason_codes:
            if rc not in unique_reason_codes:
                unique_reason_codes.append(rc)

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return ReviewResult(
            evaluation_id=f"eval_rule_{uuid.uuid4().hex[:10]}",
            response_id=response.response_id,
            quality_score=breakdown.overall_quality_score,
            confidence=0.90,
            reviewer_name=self.reviewer_name,
            reviewer_type=self.reviewer_type,
            passed_checks=list(dict.fromkeys(passed_checks)),
            failed_checks=list(dict.fromkeys(failed_checks)),
            reason_codes=unique_reason_codes,
            revision_suggestions=[],
            metadata={
                "score_breakdown": breakdown.model_dump(),
                "criteria_version": suite.version,
                "reviewer_duration_ms": round(duration_ms, 2),
            },
        )
