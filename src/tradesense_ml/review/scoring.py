"""Structured quality scoring algorithms and score breakdown data models."""

from pydantic import BaseModel, ConfigDict, Field


class QualityScoreBreakdown(BaseModel):
    """Detailed category breakdown of response quality scores."""

    model_config = ConfigDict(frozen=True)

    reasoning_quality: float = Field(
        ..., ge=0.0, le=10.0, description="Logical reasoning and analytical depth (0-10)"
    )
    coaching_usefulness: float = Field(
        ..., ge=0.0, le=10.0, description="Practical value and actionability of coaching (0-10)"
    )
    educational_value: float = Field(
        ..., ge=0.0, le=10.0, description="Pedagogical quality of educational explanations (0-10)"
    )
    consistency: float = Field(
        ..., ge=0.0, le=10.0, description="Internal alignment between scores and text (0-10)"
    )
    completeness: float = Field(
        ..., ge=0.0, le=10.0, description="Completeness of required evaluation components (0-10)"
    )
    overall_quality_score: float = Field(
        ..., ge=0.0, le=10.0, description="Weighted aggregate quality score (0-10)"
    )
    category_weights: dict[str, float] = Field(
        default_factory=dict, description="Weights applied per scoring dimension"
    )


class QualityScorer:
    """Configurable quality score calculator."""

    DEFAULT_WEIGHTS: dict[str, float] = {
        "reasoning_quality": 0.25,
        "coaching_usefulness": 0.25,
        "educational_value": 0.20,
        "consistency": 0.15,
        "completeness": 0.15,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        # Normalize weights if total > 0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def compute_score(
        self,
        reasoning_quality: float,
        coaching_usefulness: float,
        educational_value: float,
        consistency: float,
        completeness: float,
        extra_metrics: dict[str, float] | None = None,
    ) -> QualityScoreBreakdown:
        """Compute aggregate quality score from sub-scores and return detailed breakdown."""
        scores = {
            "reasoning_quality": min(max(float(reasoning_quality), 0.0), 10.0),
            "coaching_usefulness": min(max(float(coaching_usefulness), 0.0), 10.0),
            "educational_value": min(max(float(educational_value), 0.0), 10.0),
            "consistency": min(max(float(consistency), 0.0), 10.0),
            "completeness": min(max(float(completeness), 0.0), 10.0),
        }

        if extra_metrics:
            for k, v in extra_metrics.items():
                if k in scores:
                    scores[k] = min(max(float(v), 0.0), 10.0)

        weighted_sum = sum(scores[dim] * self.weights.get(dim, 0.0) for dim in scores)
        overall_score = round(weighted_sum, 2)

        return QualityScoreBreakdown(
            reasoning_quality=scores["reasoning_quality"],
            coaching_usefulness=scores["coaching_usefulness"],
            educational_value=scores["educational_value"],
            consistency=scores["consistency"],
            completeness=scores["completeness"],
            overall_quality_score=overall_score,
            category_weights=self.weights,
        )
