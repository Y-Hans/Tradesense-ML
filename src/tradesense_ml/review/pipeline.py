"""Dedicated Review Pipeline orchestrating CoachResponse quality evaluation and policy decision resolution."""

import time
from datetime import UTC, datetime
from typing import Any

from tradesense_ml.domain.schemas.coaching import CoachResponse
from tradesense_ml.domain.schemas.review import ReviewDecision, ReviewResult
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.pipelines.base import BasePipeline
from tradesense_ml.review.base import BaseReviewer
from tradesense_ml.review.criteria import ReviewCriteriaSuite
from tradesense_ml.review.decision_engine import DecisionEngine
from tradesense_ml.review.reviewers.rule_based import RuleBasedReviewer
from tradesense_ml.review.strategies import BaseReviewStrategy, SingleReviewerStrategy

logger = get_logger()


class ReviewPipeline(BasePipeline[CoachResponse, ReviewDecision]):
    """Orchestrator pipeline executing reviewer evaluation and policy decision engine resolution."""

    def __init__(
        self,
        reviewer: BaseReviewer | None = None,
        reviewers: list[BaseReviewer] | None = None,
        strategy: BaseReviewStrategy | None = None,
        criteria_suite: ReviewCriteriaSuite | None = None,
        decision_engine: DecisionEngine | None = None,
        review_version: str = "v1.0.0",
        config_version: str = "v1.0.0",
    ) -> None:
        super().__init__(pipeline_name="review_pipeline")

        # Initialize reviewers list
        if reviewers:
            self.reviewers = reviewers
        elif reviewer:
            self.reviewers = [reviewer]
        else:
            self.reviewers = [RuleBasedReviewer()]

        self.strategy = strategy or SingleReviewerStrategy()
        self.criteria_suite = criteria_suite or ReviewCriteriaSuite.default_suite()
        self.decision_engine = decision_engine or DecisionEngine()
        self.review_version = review_version
        self.config_version = config_version

    def run(self, input_data: CoachResponse, **kwargs: Any) -> ReviewDecision:
        """Execute review pipeline on a CoachResponse object.

        Workflow:
        1. Select reviewer strategy and criteria suite.
        2. Execute strategy review to produce intermediate raw ReviewResult.
        3. Pass ReviewResult to DecisionEngine to apply thresholds and render final ReviewDecision.
        4. Record execution tracking metadata.

        Args:
            input_data: Target CoachResponse payload to evaluate.
            **kwargs: Overrides for thresholds, criteria_suite, strategy, or reviewers.

        Returns:
            Structured ReviewDecision domain model.
        """
        start_time = time.perf_counter()
        logger.info(f"Starting Review Pipeline for CoachResponse '{input_data.response_id}'")

        # Runtime configuration overrides
        criteria_suite = kwargs.get("criteria_suite", self.criteria_suite)
        strategy = kwargs.get("strategy", self.strategy)
        reviewers = kwargs.get("reviewers", self.reviewers)
        engine = kwargs.get("decision_engine", self.decision_engine)

        # 1. Execute strategy-based evaluation -> intermediate ReviewResult
        eval_result: ReviewResult = strategy.execute(
            response=input_data,
            reviewers=reviewers,
            criteria_suite=criteria_suite,
            **kwargs,
        )

        # 2. Pass raw evaluation payload through policy DecisionEngine -> final ReviewDecision
        app_thresh = kwargs.get("approval_threshold")
        rev_thresh = kwargs.get("revision_threshold")
        engine_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "approval_threshold",
                "revision_threshold",
                "criteria_suite",
                "strategy",
                "reviewers",
                "decision_engine",
            )
        }

        decision: ReviewDecision = engine.evaluate_result(
            eval_result=eval_result,
            approval_threshold=app_thresh,
            revision_threshold=rev_thresh,
            **engine_kwargs,
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Attach complete execution and tracking metadata
        pipeline_metadata: dict[str, Any] = {
            "pipeline_name": self.pipeline_name,
            "review_version": self.review_version,
            "criteria_version": criteria_suite.version,
            "configuration_version": self.config_version,
            "review_strategy": (
                strategy.strategy_name
                if hasattr(strategy, "strategy_name")
                else type(strategy).__name__
            ),
            "primary_reviewer": decision.reviewer_name,
            "total_pipeline_duration_ms": round(total_latency_ms, 2),
            "processed_at": datetime.now(UTC).isoformat(),
        }

        updated_metadata = {**decision.metadata, **pipeline_metadata}

        final_decision = decision.model_copy(update={"metadata": updated_metadata})
        logger.info(
            f"Completed Review Pipeline for '{input_data.response_id}'. Verdict: {final_decision.verdict.value}, Score: {final_decision.quality_score}"
        )
        return final_decision
