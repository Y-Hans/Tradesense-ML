"""Abstract base class interface for all response reviewers."""

from abc import ABC, abstractmethod
from typing import Any

from tradesense_ml.domain.schemas.coaching import CoachResponse
from tradesense_ml.domain.schemas.review import ReviewResult


class BaseReviewer(ABC):
    """Abstract interface for CoachResponse reviewers producing raw evaluation results."""

    def __init__(self, reviewer_name: str, reviewer_type: str) -> None:
        self.reviewer_name = reviewer_name
        self.reviewer_type = reviewer_type

    @abstractmethod
    def review(
        self,
        response: CoachResponse,
        criteria_suite: Any | None = None,
        **kwargs: Any,
    ) -> ReviewResult:
        """Evaluate a CoachResponse and produce a raw ReviewResult payload.

        Args:
            response: The CoachResponse domain object to evaluate.
            criteria_suite: Optional ReviewCriteriaSuite containing evaluation parameters.
            **kwargs: Additional runtime options.

        Returns:
            ReviewResult domain model object containing raw evaluation metrics.
        """
        pass
