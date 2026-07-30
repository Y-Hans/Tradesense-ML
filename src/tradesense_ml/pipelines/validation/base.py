"""Data validation pipeline interface."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from tradesense_ml.domain.schemas.examples import ReviewedExample
from tradesense_ml.pipelines.base import BasePipeline


class ValidationResult(BaseModel):
    """Validation report."""

    is_valid: bool = Field(..., description="Validation outcome")
    errors: list[str] = Field(default_factory=list, description="Validation failure messages")
    warnings: list[str] = Field(default_factory=list, description="Validation warning messages")


class BaseValidatorPipeline(BasePipeline[ReviewedExample, ValidationResult], ABC):
    """Automated schema, rule, and syntax validator stage."""

    def __init__(self) -> None:
        super().__init__(pipeline_name="validation_pipeline")

    @abstractmethod
    def run(self, input_data: ReviewedExample, **kwargs: Any) -> ValidationResult:
        """Validate dataset example."""
        pass
