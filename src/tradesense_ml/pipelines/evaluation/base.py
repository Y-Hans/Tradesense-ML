"""Evaluation and benchmark runner pipeline interfaces."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from tradesense_ml.domain.schemas.examples import EvaluationExample
from tradesense_ml.pipelines.base import BasePipeline


class BenchmarkSummary(BaseModel):
    """Summary of benchmark run results across all samples."""

    benchmark_id: str = Field(..., description="Benchmark suite ID")
    model_id: str = Field(..., description="Model under test ID")
    json_validity_rate: float = Field(..., ge=0.0, le=1.0)
    reason_code_precision: float = Field(..., ge=0.0, le=1.0)
    risk_explanation_score: float = Field(..., ge=0.0, le=10.0)
    discipline_explanation_score: float = Field(..., ge=0.0, le=10.0)
    hallucination_rate: float = Field(..., ge=0.0, le=1.0)
    average_latency_ms: float = Field(..., ge=0.0)
    detailed_results: list[dict[str, Any]] = Field(default_factory=list)


class BaseEvaluationPipeline(BasePipeline[list[EvaluationExample], BenchmarkSummary], ABC):
    """Abstract benchmark runner pipeline."""

    def __init__(self) -> None:
        super().__init__(pipeline_name="evaluation_pipeline")

    @abstractmethod
    def run(self, input_data: list[EvaluationExample], **kwargs: Any) -> BenchmarkSummary:
        """Run benchmark suite against target model."""
        pass
