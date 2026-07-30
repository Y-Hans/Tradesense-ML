"""Dedicated 4-stage Review Pipeline Architecture covering:
1. Automated Validation Stage
2. AI Teacher Review Stage
3. Human Review Stage
4. Approval Stage
"""

from abc import ABC, abstractmethod
from typing import Any

from tradesense_ml.domain.schemas.examples import ReviewedExample
from tradesense_ml.domain.schemas.review import (
    ReviewAuditRecord,
    ReviewDecision,
    ReviewStage,
)
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.pipelines.base import BasePipeline

logger = get_logger()


class BaseReviewStage(ABC):
    """Abstract stage in the review pipeline."""

    def __init__(self, stage_enum: ReviewStage) -> None:
        self.stage_enum = stage_enum

    @abstractmethod
    def process(self, example: ReviewedExample) -> tuple[ReviewedExample, ReviewAuditRecord]:
        """Execute stage processing and return updated example and audit record."""
        pass


class ReviewPipeline(BasePipeline[ReviewedExample, ReviewedExample]):
    """Orchestrator for multi-stage review pipeline."""

    def __init__(self, stages: list[BaseReviewStage] | None = None) -> None:
        super().__init__(pipeline_name="review_pipeline")
        self.stages = stages or []

    def add_stage(self, stage: BaseReviewStage) -> None:
        """Append review stage to pipeline."""
        self.stages.append(stage)

    def run(self, input_data: ReviewedExample, **kwargs: Any) -> ReviewedExample:
        """Run input example sequentially through all review stages."""
        current_example = input_data
        logger.info(f"Starting Review Pipeline for example {current_example.example_id}")

        for stage in self.stages:
            logger.info(
                f"Executing stage '{stage.stage_enum.value}' on {current_example.example_id}"
            )
            current_example, audit_record = stage.process(current_example)

            # Append audit record to immutable history
            updated_audit = list(current_example.audit_trail) + [audit_record]
            current_example = current_example.model_copy(
                update={"review_status": stage.stage_enum, "audit_trail": updated_audit}
            )

            if audit_record.decision == ReviewDecision.REJECT:
                logger.warning(
                    f"Example {current_example.example_id} rejected at stage '{stage.stage_enum.value}'"
                )
                return current_example.model_copy(update={"review_status": ReviewStage.REJECTED})

        logger.info(f"Example {current_example.example_id} successfully passed all review stages.")
        return current_example.model_copy(update={"review_status": ReviewStage.APPROVED})
