"""Dataset validation engine enforcing schema compliance, field completeness, and split integrity."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.dataset import DatasetExample
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class ValidationReport(BaseModel):
    """Report payload detailing dataset validation results."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(..., description="Whether dataset passed all validation rules")
    total_evaluated: int = Field(..., description="Number of examples evaluated")
    errors: list[str] = Field(default_factory=list, description="List of error messages")
    warnings: list[str] = Field(default_factory=list, description="List of warning messages")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Validation metadata summary"
    )


class DatasetValidator:
    """Validator ensuring dataset structural, schema, lineage, and split integrity."""

    def validate_dataset(
        self,
        examples: list[DatasetExample],
        split_dict: dict[str, list[DatasetExample]] | None = None,
        require_review_info: bool = True,
        require_lineage: bool = True,
    ) -> ValidationReport:
        """Validate a list of DatasetExample records and optional split mapping.

        Args:
            examples: List of DatasetExample records.
            split_dict: Optional dictionary mapping split names to lists of DatasetExample objects.
            require_review_info: Whether to enforce presence of review_info.
            require_lineage: Whether to enforce presence of lineage metadata.

        Returns:
            ValidationReport detailing validation pass/fail status and error list.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not examples:
            errors.append("Dataset is empty; contains 0 examples.")
            return ValidationReport(
                is_valid=False, total_evaluated=0, errors=errors, warnings=warnings
            )

        seen_ids: set[str] = set()

        for idx, ex in enumerate(examples):
            prefix = f"Example #{idx+1} ({ex.example_id}):"

            # 1. Required fields check
            if not ex.example_id or not ex.example_id.strip():
                errors.append(f"Example #{idx+1}: Missing required example_id.")
            if not ex.instruction or not ex.instruction.strip():
                errors.append(f"{prefix} Empty or missing 'instruction'.")
            if not ex.input or not ex.input.strip():
                errors.append(f"{prefix} Empty or missing 'input'.")
            if not ex.output or not ex.output.strip():
                errors.append(f"{prefix} Empty or missing 'output'.")
            if not ex.prompt or not ex.prompt.strip():
                errors.append(f"{prefix} Empty or missing 'prompt'.")

            # 2. Duplicate IDs check
            if ex.example_id in seen_ids:
                errors.append(f"{prefix} Duplicate example_id '{ex.example_id}' detected.")
            else:
                seen_ids.add(ex.example_id)

            # 3. Chat format messages validation if present
            if ex.messages:
                roles = [m.get("role") for m in ex.messages if isinstance(m, dict)]
                if "user" not in roles or "assistant" not in roles:
                    warnings.append(
                        f"{prefix} Chat messages list missing 'user' or 'assistant' role."
                    )

            # 4. Review info check
            if require_review_info and not ex.review_info:
                warnings.append(f"{prefix} Missing review_info metadata dictionary.")

            # 5. Lineage metadata check
            if require_lineage and not ex.lineage:
                warnings.append(f"{prefix} Missing provenance lineage metadata.")

        # 6. Split integrity check (no overlapping IDs between splits)
        if split_dict:
            split_seen: dict[str, set[str]] = {}
            for split_name, split_list in split_dict.items():
                split_seen[split_name] = {e.example_id for e in split_list}

            split_names = list(split_dict.keys())
            for i in range(len(split_names)):
                for j in range(i + 1, len(split_names)):
                    name_i, name_j = split_names[i], split_names[j]
                    overlap = split_seen[name_i].intersection(split_seen[name_j])
                    if overlap:
                        errors.append(
                            f"Split integrity error: {len(overlap)} overlapping IDs between split '{name_i}' and '{name_j}'."
                        )

        is_valid = len(errors) == 0
        if is_valid:
            logger.info(f"Dataset validation PASSED cleanly for {len(examples)} examples.")
        else:
            logger.error(
                f"Dataset validation FAILED with {len(errors)} errors and {len(warnings)} warnings."
            )

        return ValidationReport(
            is_valid=is_valid,
            total_evaluated=len(examples),
            errors=errors,
            warnings=warnings,
            metadata={"unique_examples": len(seen_ids)},
        )
