"""Retry Strategy module providing provider-agnostic retry capabilities with exponential backoff."""

import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.logging.logger import get_logger
from tradesense_ml.teachers.response_parser import ResponseParsingError
from tradesense_ml.teachers.validator import ResponseValidationError

logger = get_logger()

T = TypeVar("T")


class RetryConfig(BaseModel):
    """Configuration options for retry execution."""

    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(default=3, ge=0, description="Maximum number of retry attempts")
    initial_backoff_sec: float = Field(default=0.5, ge=0.0, description="Initial backoff delay in seconds")
    backoff_factor: float = Field(default=2.0, ge=1.0, description="Exponential backoff multiplier")
    retry_on_parsing_error: bool = Field(
        default=True, description="Whether to retry on JSON/parsing errors"
    )
    retry_on_validation_error: bool = Field(
        default=True, description="Whether to retry on validation failures"
    )


class RetryHandler:
    """Provider-agnostic retry handler executing callables with configurable exponential backoff."""

    def __init__(self, config: RetryConfig | None = None) -> None:
        self.config = config or RetryConfig()

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute callable with retries on transient errors, parsing failures, or validation errors."""
        retries = 0
        backoff = self.config.initial_backoff_sec

        while True:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                retries += 1
                should_retry = self._should_retry(exc)

                if not should_retry or retries > self.config.max_retries:
                    logger.error(
                        f"Retry limit exceeded ({retries}/{self.config.max_retries}) or non-retryable exception: {exc}"
                    )
                    raise exc

                logger.warning(
                    f"Attempt {retries}/{self.config.max_retries} failed with error: {exc}. Retrying in {backoff:.2f}s..."
                )
                time.sleep(backoff)
                backoff *= self.config.backoff_factor

    def _should_retry(self, exc: Exception) -> bool:
        """Determine if an exception qualifies for retry based on configuration."""
        if isinstance(exc, ResponseParsingError):
            return self.config.retry_on_parsing_error
        if isinstance(exc, ResponseValidationError):
            return self.config.retry_on_validation_error
        # Treat network, timeout, or runtime exceptions as transient retryable errors
        if isinstance(exc, (TimeoutError, ConnectionError, RuntimeError, OSError)):
            return True
        return False
