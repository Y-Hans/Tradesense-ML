"""Review execution strategies supporting single-reviewer and extension hooks."""

from abc import ABC, abstractmethod
from typing import Any

from tradesense_ml.domain.schemas.coaching import CoachResponse
from tradesense_ml.domain.schemas.review import ReviewResult
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.review.base import BaseReviewer
from tradesense_ml.review.criteria import ReviewCriteriaSuite

logger = get_logger()


class BaseReviewStrategy(ABC):
    """Abstract review strategy for orchestrating reviewer execution."""

    def __init__(self, strategy_name: str) -> None:
        self.strategy_name = strategy_name

    @abstractmethod
    def execute(
        self,
        response: CoachResponse,
        reviewers: list[BaseReviewer],
        criteria_suite: ReviewCriteriaSuite,
        **kwargs: Any,
    ) -> ReviewResult:
        """Execute review using strategy workflow and return raw evaluation ReviewResult."""
        pass


class SingleReviewerStrategy(BaseReviewStrategy):
    """Strategy that delegates response evaluation to a single primary reviewer."""

    def __init__(self) -> None:
        super().__init__(strategy_name="single")

    def execute(
        self,
        response: CoachResponse,
        reviewers: list[BaseReviewer],
        criteria_suite: ReviewCriteriaSuite,
        **kwargs: Any,
    ) -> ReviewResult:
        """Execute evaluation using the first available reviewer."""
        if not reviewers:
            raise ValueError(
                "SingleReviewerStrategy requires at least one reviewer in reviewers list."
            )

        reviewer = reviewers[0]
        logger.info(
            f"Executing SingleReviewerStrategy using reviewer '{reviewer.reviewer_name}' on response '{response.response_id}'"
        )

        result = reviewer.review(response=response, criteria_suite=criteria_suite, **kwargs)

        # Attach strategy metadata
        updated_meta = {**result.metadata, "review_strategy": self.strategy_name}
        return result.model_copy(update={"metadata": updated_meta})


class MultiReviewerStrategy(BaseReviewStrategy):
    """Extension hook for parallel multi-reviewer execution (Scaffold for future milestone)."""

    def __init__(self) -> None:
        super().__init__(strategy_name="multi")

    def execute(
        self,
        response: CoachResponse,
        reviewers: list[BaseReviewer],
        criteria_suite: ReviewCriteriaSuite,
        **kwargs: Any,
    ) -> ReviewResult:
        """Extension point scaffold for multi-reviewer execution."""
        raise NotImplementedError(
            "MultiReviewerStrategy is an extension hook reserved for future milestones."
        )


class ConsensusStrategy(BaseReviewStrategy):
    """Extension hook for voting consensus review aggregation (Scaffold for future milestone)."""

    def __init__(self) -> None:
        super().__init__(strategy_name="consensus")

    def execute(
        self,
        response: CoachResponse,
        reviewers: list[BaseReviewer],
        criteria_suite: ReviewCriteriaSuite,
        **kwargs: Any,
    ) -> ReviewResult:
        """Extension point scaffold for consensus aggregation."""
        raise NotImplementedError(
            "ConsensusStrategy is an extension hook reserved for future milestones."
        )


class DebateStrategy(BaseReviewStrategy):
    """Extension hook for multi-agent review debate resolution (Scaffold for future milestone)."""

    def __init__(self) -> None:
        super().__init__(strategy_name="debate")

    def execute(
        self,
        response: CoachResponse,
        reviewers: list[BaseReviewer],
        criteria_suite: ReviewCriteriaSuite,
        **kwargs: Any,
    ) -> ReviewResult:
        """Extension point scaffold for debate resolution."""
        raise NotImplementedError(
            "DebateStrategy is an extension hook reserved for future milestones."
        )
