"""Structured Response Parser module for converting raw LLM output into validated CoachResponse models."""

import json
import re
from typing import Any

from pydantic import ValidationError

from tradesense_ml.domain.schemas.coaching import CoachResponse
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class ResponseParsingError(Exception):
    """Exception raised when raw LLM provider output cannot be parsed into a CoachResponse."""

    def __init__(self, message: str, raw_content: str, details: Any = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content
        self.details = details


class ResponseParser:
    """Parser to transform raw LLM text or JSON payloads into validated CoachResponse schemas."""

    @staticmethod
    def parse(raw_output: str | dict[str, Any], request_id: str) -> CoachResponse:
        """Parse raw text or dict provider response into a CoachResponse object.

        Handles markdown-wrapped JSON, missing wrappers, and field normalization.
        """
        payload_dict: dict[str, Any]

        if isinstance(raw_output, dict):
            payload_dict = raw_output
        elif isinstance(raw_output, str):
            cleaned = ResponseParser._extract_json_string(raw_output)
            try:
                payload_dict = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                logger.error(f"Failed to parse JSON for request {request_id}: {exc}")
                raise ResponseParsingError(
                    f"Malformed JSON output: {exc}", raw_content=raw_output, details=str(exc)
                ) from exc
        else:
            raise ResponseParsingError(
                f"Unsupported output type '{type(raw_output).__name__}'",
                raw_content=str(raw_output),
            )

        # Unwrap if nested under outer container keys
        if "coach_response" in payload_dict and isinstance(payload_dict["coach_response"], dict):
            payload_dict = payload_dict["coach_response"]
        elif "response" in payload_dict and isinstance(payload_dict["response"], dict):
            payload_dict = payload_dict["response"]

        # Ensure required identifiers are populated
        if "request_id" not in payload_dict or not payload_dict["request_id"]:
            payload_dict["request_id"] = request_id
        if "response_id" not in payload_dict or not payload_dict["response_id"]:
            payload_dict["response_id"] = f"resp_{request_id}"

        # Standardize actionable_advice if passed as a single string
        if "actionable_advice" in payload_dict and isinstance(
            payload_dict["actionable_advice"], str
        ):
            lines = [
                line.lstrip("-*• ").strip()
                for line in payload_dict["actionable_advice"].splitlines()
                if line.strip()
            ]
            payload_dict["actionable_advice"] = (
                lines if lines else [payload_dict["actionable_advice"]]
            )

        try:
            return CoachResponse.model_validate(payload_dict)
        except ValidationError as exc:
            logger.error(
                f"Validation error during response parsing for request {request_id}: {exc}"
            )
            raise ResponseParsingError(
                f"Schema validation failure during parsing: {exc}",
                raw_content=str(raw_output),
                details=exc.errors(),
            ) from exc

    @staticmethod
    def _extract_json_string(content: str) -> str:
        """Extract valid JSON string from raw text, handling markdown codeblocks."""
        stripped = content.strip()

        # Match markdown ```json ... ``` or ``` ... ```
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(pattern, stripped, re.IGNORECASE)
        if matches:
            return str(matches[0]).strip()

        # If no codeblock, locate first '{' and last '}'
        start_idx = stripped.find("{")
        end_idx = stripped.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return stripped[start_idx : end_idx + 1]

        return stripped
