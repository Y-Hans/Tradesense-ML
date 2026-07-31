"""Response Validator module for enforcing domain schema compliance and quality criteria on CoachResponse objects."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.coaching import CoachResponse
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class ValidationResult(BaseModel):
    """Result payload of a CoachResponse validation check."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(..., description="Whether response passed all validation rules")
    errors: list[str] = Field(default_factory=list, description="List of validation error messages")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Audit validation metadata")


class ResponseValidationError(Exception):
    """Exception raised when a CoachResponse fails business validation rules."""

    def __init__(self, message: str, errors: list[str]) -> None:
        super().__init__(message)
        self.errors = errors


class ResponseValidator:
    """Validator that checks generated CoachResponse objects against business rules and boundary constraints."""

    @staticmethod
    def validate(response: CoachResponse) -> ValidationResult:
        """Validate a CoachResponse object and return a structured ValidationResult."""
        errors: list[str] = []

        # 1. Headline validation
        if not response.headline or not response.headline.strip():
            errors.append("Headline is missing or empty.")

        # 2. Score boundary validations
        if not (0.0 <= response.overall_score <= 10.0):
            errors.append(f"overall_score {response.overall_score} out of bounds [0.0, 10.0].")

        risk_eval = response.risk_evaluation
        if not (0.0 <= risk_eval.risk_score <= 10.0):
            errors.append(f"risk_score {risk_eval.risk_score} out of bounds [0.0, 10.0].")
        if not risk_eval.risk_summary or not risk_eval.risk_summary.strip():
            errors.append("risk_summary is missing or empty.")

        disc_eval = response.discipline_evaluation
        if not (0.0 <= disc_eval.discipline_score <= 10.0):
            errors.append(f"discipline_score {disc_eval.discipline_score} out of bounds [0.0, 10.0].")
        if not (0.0 <= disc_eval.plan_adherence_score <= 10.0):
            errors.append(f"plan_adherence_score {disc_eval.plan_adherence_score} out of bounds [0.0, 10.0].")
        if not disc_eval.discipline_summary or not disc_eval.discipline_summary.strip():
            errors.append("discipline_summary is missing or empty.")

        # 3. Actionable advice validation
        if not response.actionable_advice:
            errors.append("actionable_advice list is empty.")
        else:
            empty_items = [item for item in response.actionable_advice if not str(item).strip()]
            if empty_items:
                errors.append("actionable_advice contains empty or whitespace-only items.")

        # 4. Educational note validation
        if not response.educational_note or not response.educational_note.strip():
            errors.append("educational_note is missing or empty.")

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Response {response.response_id} failed validation with {len(errors)} error(s): {errors}")

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            metadata={
                "response_id": response.response_id,
                "request_id": response.request_id,
            },
        )

    @classmethod
    def validate_and_raise(cls, response: CoachResponse) -> None:
        """Validate a CoachResponse object and raise ResponseValidationError if invalid."""
        result = cls.validate(response)
        if not result.is_valid:
            raise ResponseValidationError(
                f"CoachResponse failed validation with {len(result.errors)} error(s).",
                errors=result.errors,
            )
