"""Reason code generation and revision suggestion mapping utilities."""

from tradesense_ml.domain.schemas.review import ReasonCode, ReviewVerdict


class RevisionSuggestionGenerator:
    """Helper for generating structured, actionable revision suggestions."""

    REASON_SUGGESTION_MAP: dict[ReasonCode, str] = {
        ReasonCode.INCONSISTENT_REASONING: (
            "Align overall score with individual risk and discipline evaluation summaries."
        ),
        ReasonCode.MISSING_ACTIONABLE_ADVICE: (
            "Provide more actionable advice with specific, measurable trading steps."
        ),
        ReasonCode.LOW_EDUCATIONAL_VALUE: (
            "Improve explanation of trading concepts and risk management principles."
        ),
        ReasonCode.HALLUCINATED_MARKET_FACT: (
            "Remove unverified or contradictory market statements."
        ),
        ReasonCode.INSUFFICIENT_EXPLANATION: (
            "Explain why discipline and risk scores were assigned in greater detail."
        ),
        ReasonCode.UNSAFE_CONTENT: (
            "Remove unsafe trade recommendations or reckless financial advice."
        ),
        ReasonCode.STYLE_VIOLATION: (
            "Reduce generic statements and adhere to concise coaching tone."
        ),
        ReasonCode.INCOMPLETE_RESPONSE: (
            "Complete all required fields in risk, discipline, and actionable advice sections."
        ),
    }

    @classmethod
    def generate_suggestions(
        cls,
        failed_checks: list[str],
        reason_codes: list[ReasonCode],
        verdict: ReviewVerdict,
    ) -> list[str]:
        """Generate structured revision suggestions based on failed checks and reason codes."""
        if verdict == ReviewVerdict.APPROVE:
            return []

        suggestions: list[str] = []

        # Map reason codes to specific suggestion text
        for code in reason_codes:
            if code in cls.REASON_SUGGESTION_MAP:
                sugg = cls.REASON_SUGGESTION_MAP[code]
                if sugg not in suggestions:
                    suggestions.append(sugg)

        # Fallback suggestions based on failed checks if suggestions list is still empty
        if not suggestions:
            for check in failed_checks:
                if "risk" in check and "Improve explanation of risk management." not in suggestions:
                    suggestions.append("Improve explanation of risk management.")
                elif (
                    "discipline" in check
                    and "Explain why discipline score was assigned." not in suggestions
                ):
                    suggestions.append("Explain why discipline score was assigned.")
                elif (
                    "actionability" in check
                    and "Provide more actionable advice." not in suggestions
                ):
                    suggestions.append("Provide more actionable advice.")
                elif "style" in check and "Reduce generic statements." not in suggestions:
                    suggestions.append("Reduce generic statements.")

        if not suggestions:
            suggestions.append(
                "Revise coaching response to address identified quality deficiencies."
            )

        return suggestions
