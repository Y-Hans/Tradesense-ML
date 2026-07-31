"""Configurable review criteria definitions and default criteria suites."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReviewCriterion(BaseModel):
    """Definition of a single review criterion."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique criterion identifier")
    description: str = Field(..., description="Description of the criterion check")
    weight: float = Field(
        default=1.0, ge=0.0, le=10.0, description="Criterion weight in quality scoring"
    )
    threshold: float = Field(
        default=6.0, ge=0.0, le=10.0, description="Minimum passing score for this criterion"
    )
    enabled: bool = Field(default=True, description="Whether this criterion check is active")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Criterion-specific metadata"
    )


class ReviewCriteriaSuite(BaseModel):
    """Collection of configurable review criteria."""

    model_config = ConfigDict(frozen=True)

    suite_id: str = Field(default="default_suite", description="Suite identifier")
    version: str = Field(default="v1.0.0", description="Criteria version string")
    criteria: dict[str, ReviewCriterion] = Field(
        default_factory=dict, description="Map of criterion name to definition"
    )

    @classmethod
    def default_suite(cls, version: str = "v1.0.0") -> "ReviewCriteriaSuite":
        """Construct the default standard criteria suite covering all required dimensions."""
        standard_criteria = [
            ReviewCriterion(
                name="coaching_quality",
                description="Headline clarity, tone, and overall coaching message relevance",
                weight=1.5,
                threshold=6.5,
            ),
            ReviewCriterion(
                name="risk_analysis_quality",
                description="Accuracy and completeness of risk evaluation scores and summary",
                weight=1.5,
                threshold=6.5,
            ),
            ReviewCriterion(
                name="discipline_analysis_quality",
                description="Evaluation of FOMO, overtrading, revenge trading, and plan adherence",
                weight=1.5,
                threshold=6.5,
            ),
            ReviewCriterion(
                name="internal_consistency",
                description="Logical consistency between sub-scores, summary text, and overall score",
                weight=1.2,
                threshold=6.0,
            ),
            ReviewCriterion(
                name="educational_value",
                description="Substance, explanation quality, and pedagogical value of educational notes",
                weight=1.0,
                threshold=6.0,
            ),
            ReviewCriterion(
                name="actionability",
                description="Concrete, actionable, and non-generic nature of provided advice points",
                weight=1.2,
                threshold=6.0,
            ),
            ReviewCriterion(
                name="completeness",
                description="Presence of all mandatory response fields and non-empty section text",
                weight=1.0,
                threshold=7.0,
            ),
            ReviewCriterion(
                name="factual_consistency",
                description="Absence of fabricated market terminology, inaccurate facts, or internal self-contradictions",
                weight=1.0,
                threshold=7.0,
            ),
            ReviewCriterion(
                name="style_consistency",
                description="Formatting compliance, absence of excessive jargon or generic fluff",
                weight=0.8,
                threshold=5.5,
            ),
            ReviewCriterion(
                name="safety",
                description="Absence of harmful advice, profanity, or illegal trading recommendations",
                weight=1.0,
                threshold=8.0,
            ),
        ]

        criteria_map = {c.name: c for c in standard_criteria}
        return cls(suite_id="tradesense_standard_v1", version=version, criteria=criteria_map)
