"""Synthetic dataset validator ensuring schema adherence and mathematical consistency."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from tradesense_ml.domain.schemas.coaching import CoachRequest
from tradesense_ml.domain.schemas.trade import Side, Trade


@dataclass
class ValidationResult:
    """Container for validation results of a dataset sample."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SyntheticDatasetValidator:
    """Validator for synthetic trading and market dataset records."""

    def validate_sample(self, sample: CoachRequest | dict[str, Any]) -> ValidationResult:
        """Validate a single synthetic CoachRequest or raw dictionary sample."""
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Schema Validation
        if isinstance(sample, dict):
            try:
                coach_request = CoachRequest.model_validate(sample)
            except ValidationError as e:
                return ValidationResult(
                    is_valid=False, errors=[f"Schema validation error: {str(e)}"]
                )
        else:
            coach_request = sample

        trade: Trade = coach_request.trade

        # 2. Basic Value Bounds
        if trade.entry_price <= 0:
            errors.append(f"Invalid entry price: {trade.entry_price} must be > 0")
        if trade.quantity <= 0:
            errors.append(f"Invalid quantity: {trade.quantity} must be > 0")
        if trade.exit_price is not None and trade.exit_price <= 0:
            errors.append(f"Invalid exit price: {trade.exit_price} must be > 0")

        # 3. Timestamps check
        if trade.exit_timestamp is not None and trade.entry_timestamp > trade.exit_timestamp:
            errors.append(
                f"Entry timestamp ({trade.entry_timestamp}) is after exit timestamp ({trade.exit_timestamp})"
            )

        # 4. Executions check
        if trade.executions:
            exit_execs = [e for e in trade.executions if e.execution_id.startswith("exec_out")]
            if exit_execs:
                total_exit_qty = sum(e.quantity for e in exit_execs)
                if abs(total_exit_qty - trade.quantity) > 1e-3:
                    errors.append(
                        f"Execution quantity mismatch: total exit fills sum {total_exit_qty} != position quantity {trade.quantity}"
                    )

        # 5. Mathematical PnL Consistency
        if trade.exit_price is not None and trade.pnl is not None:
            is_long = trade.side in (Side.LONG, Side.BUY)
            raw_pnl = (
                (trade.exit_price - trade.entry_price) * trade.quantity
                if is_long
                else (trade.entry_price - trade.exit_price) * trade.quantity
            )
            total_fees = sum(e.fee for e in trade.executions) if trade.executions else 0.0
            expected_pnl = round(raw_pnl - total_fees, 2)

            if abs(trade.pnl - expected_pnl) > 0.10:
                errors.append(
                    f"PnL math inconsistency: recorded PnL {trade.pnl} != expected PnL {expected_pnl} (raw={raw_pnl:.2f}, fees={total_fees:.2f})"
                )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_batch(
        self, samples: list[CoachRequest] | list[dict[str, Any]]
    ) -> tuple[bool, list[ValidationResult]]:
        """Validate a batch of synthetic dataset samples. Return overall status and results."""
        results = [self.validate_sample(sample) for sample in samples]
        all_valid = all(r.is_valid for r in results)
        return all_valid, results
