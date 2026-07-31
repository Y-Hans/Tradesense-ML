"""Extension scaffolds for future AI, Consensus, and Human Reviewers."""

import uuid
from typing import Any

from tradesense_ml.domain.schemas.coaching import CoachResponse
from tradesense_ml.domain.schemas.review import ReasonCode, ReviewResult
from tradesense_ml.review.base import BaseReviewer
from tradesense_ml.review.criteria import ReviewCriteriaSuite


class GPTReviewer(BaseReviewer):
    """Extension scaffold for GPT-based LLM response reviewer."""

    def __init__(self, model_name: str = "gpt-4o") -> None:
        super().__init__(reviewer_name=f"gpt_reviewer_{model_name}", reviewer_type="ai_teacher")
        self.model_name = model_name

    def review(
        self,
        response: CoachResponse,
        criteria_suite: ReviewCriteriaSuite | None = None,
        **kwargs: Any,
    ) -> ReviewResult:
        """Mock AI review execution for testing / extension."""
        quality_score = float(kwargs.get("quality_score", response.overall_score))
        return ReviewResult(
            evaluation_id=f"eval_gpt_{uuid.uuid4().hex[:8]}",
            response_id=response.response_id,
            quality_score=quality_score,
            confidence=0.95,
            reviewer_name=self.reviewer_name,
            reviewer_type=self.reviewer_type,
            passed_checks=["coaching_quality", "risk_analysis_quality", "safety"],
            failed_checks=[] if quality_score >= 7.0 else ["internal_consistency"],
            reason_codes=(
                [ReasonCode.GOOD_RISK_ANALYSIS]
                if quality_score >= 7.0
                else [ReasonCode.INCONSISTENT_REASONING]
            ),
            revision_suggestions=[],
            metadata={"model_name": self.model_name, "mock": True, "reviewer_duration_ms": 450.0},
        )


class ClaudeReviewer(BaseReviewer):
    """Extension scaffold for Claude-based LLM response reviewer."""

    def __init__(self, model_name: str = "claude-3-5-sonnet") -> None:
        super().__init__(reviewer_name=f"claude_reviewer_{model_name}", reviewer_type="ai_teacher")
        self.model_name = model_name

    def review(
        self,
        response: CoachResponse,
        criteria_suite: ReviewCriteriaSuite | None = None,
        **kwargs: Any,
    ) -> ReviewResult:
        """Mock AI review execution for testing / extension."""
        quality_score = float(kwargs.get("quality_score", response.overall_score))
        return ReviewResult(
            evaluation_id=f"eval_claude_{uuid.uuid4().hex[:8]}",
            response_id=response.response_id,
            quality_score=quality_score,
            confidence=0.94,
            reviewer_name=self.reviewer_name,
            reviewer_type=self.reviewer_type,
            passed_checks=["coaching_quality", "discipline_analysis_quality", "educational_value"],
            failed_checks=[],
            reason_codes=[ReasonCode.EXCELLENT_COACHING],
            revision_suggestions=[],
            metadata={"model_name": self.model_name, "mock": True, "reviewer_duration_ms": 410.0},
        )


class GeminiReviewer(BaseReviewer):
    """Extension scaffold for Gemini-based LLM response reviewer."""

    def __init__(self, model_name: str = "gemini-1.5-pro") -> None:
        super().__init__(reviewer_name=f"gemini_reviewer_{model_name}", reviewer_type="ai_teacher")
        self.model_name = model_name

    def review(
        self,
        response: CoachResponse,
        criteria_suite: ReviewCriteriaSuite | None = None,
        **kwargs: Any,
    ) -> ReviewResult:
        """Mock AI review execution for testing / extension."""
        quality_score = float(kwargs.get("quality_score", response.overall_score))
        return ReviewResult(
            evaluation_id=f"eval_gemini_{uuid.uuid4().hex[:8]}",
            response_id=response.response_id,
            quality_score=quality_score,
            confidence=0.92,
            reviewer_name=self.reviewer_name,
            reviewer_type=self.reviewer_type,
            passed_checks=["coaching_quality", "actionability", "completeness"],
            failed_checks=[],
            reason_codes=[ReasonCode.GOOD_ACTION_PLAN],
            revision_suggestions=[],
            metadata={"model_name": self.model_name, "mock": True, "reviewer_duration_ms": 380.0},
        )


class ConsensusReviewer(BaseReviewer):
    """Extension scaffold for consensus aggregation reviewer."""

    def __init__(self) -> None:
        super().__init__(reviewer_name="consensus_reviewer", reviewer_type="consensus")

    def review(
        self,
        response: CoachResponse,
        criteria_suite: ReviewCriteriaSuite | None = None,
        **kwargs: Any,
    ) -> ReviewResult:
        """Mock consensus review execution for testing / extension."""
        return ReviewResult(
            evaluation_id=f"eval_cons_{uuid.uuid4().hex[:8]}",
            response_id=response.response_id,
            quality_score=8.5,
            confidence=0.98,
            reviewer_name=self.reviewer_name,
            reviewer_type=self.reviewer_type,
            passed_checks=["coaching_quality", "safety", "internal_consistency"],
            failed_checks=[],
            reason_codes=[ReasonCode.EXCELLENT_COACHING],
            revision_suggestions=[],
            metadata={"mock": True, "reviewer_duration_ms": 950.0},
        )


class HumanReviewer(BaseReviewer):
    """Extension scaffold for human expert reviewer integration."""

    def __init__(self, reviewer_id: str = "human_expert_01") -> None:
        super().__init__(reviewer_name=reviewer_id, reviewer_type="human")

    def review(
        self,
        response: CoachResponse,
        criteria_suite: ReviewCriteriaSuite | None = None,
        **kwargs: Any,
    ) -> ReviewResult:
        """Mock human review execution for testing / extension."""
        return ReviewResult(
            evaluation_id=f"eval_hum_{uuid.uuid4().hex[:8]}",
            response_id=response.response_id,
            quality_score=float(kwargs.get("score", 9.0)),
            confidence=1.0,
            reviewer_name=self.reviewer_name,
            reviewer_type=self.reviewer_type,
            passed_checks=["all"],
            failed_checks=[],
            reason_codes=[ReasonCode.EXCELLENT_COACHING],
            revision_suggestions=[],
            metadata={
                "comments": kwargs.get("comments", "Human approved"),
                "reviewer_duration_ms": 0.0,
            },
        )
