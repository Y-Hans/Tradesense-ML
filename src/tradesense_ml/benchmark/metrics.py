"""Reusable metric abstractions and metric registry for benchmark suite evaluations."""

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from tradesense_ml.domain.schemas.benchmark import BenchmarkMetric


class BaseBenchmarkMetric(ABC):
    """Abstract base class for independent benchmark metric calculators."""

    metric_id: str
    name: str
    metric_type: str
    unit: str
    min_value: float = 0.0
    max_value: float = 10.0

    @abstractmethod
    def compute(
        self,
        predictions: list[Any],
        targets: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkMetric:
        """Compute metric value and return canonical BenchmarkMetric."""
        pass


class MetricRegistry:
    """Central registry for discovering and instantiating benchmark metrics."""

    _registry: ClassVar[dict[str, type[BaseBenchmarkMetric]]] = {}

    @classmethod
    def register(cls, metric_cls: type[BaseBenchmarkMetric]) -> type[BaseBenchmarkMetric]:
        """Register metric class by metric_id."""
        cls._registry[metric_cls.metric_id] = metric_cls
        return metric_cls

    @classmethod
    def get(cls, metric_id: str) -> BaseBenchmarkMetric:
        """Instantiate registered metric by metric_id."""
        if metric_id not in cls._registry:
            raise KeyError(
                f"Metric '{metric_id}' not found in MetricRegistry. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[metric_id]()

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List registered metric IDs."""
        return list(cls._registry.keys())


@MetricRegistry.register
class AccuracyMetric(BaseBenchmarkMetric):
    """Calculates accuracy metric (proportion of matching outputs)."""

    metric_id = "accuracy"
    name = "Accuracy"
    metric_type = "accuracy"
    unit = "%"
    min_value = 0.0
    max_value = 1.0

    def compute(
        self,
        predictions: list[Any],
        targets: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkMetric:
        if not predictions:
            val = 0.0
        elif targets and len(targets) == len(predictions):
            matches = sum(1 for p, t in zip(predictions, targets) if p == t)
            val = matches / len(predictions)
        else:
            # Fallback: check truthiness of predictions
            val = sum(1 for p in predictions if bool(p)) / len(predictions)

        return BenchmarkMetric(
            metric_id=self.metric_id,
            name=self.name,
            metric_type=self.metric_type,
            value=round(val, 4),
            unit=self.unit,
            min_value=self.min_value,
            max_value=self.max_value,
            metadata=metadata or {},
        )


@MetricRegistry.register
class PassRateMetric(BaseBenchmarkMetric):
    """Calculates pass rate percentage across evaluated samples."""

    metric_id = "pass_rate"
    name = "Pass Rate"
    metric_type = "pass_rate"
    unit = "%"
    min_value = 0.0
    max_value = 1.0

    def compute(
        self,
        predictions: list[Any],
        targets: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkMetric:
        if not predictions:
            val = 0.0
        else:
            val = sum(1 for p in predictions if bool(p)) / len(predictions)

        return BenchmarkMetric(
            metric_id=self.metric_id,
            name=self.name,
            metric_type=self.metric_type,
            value=round(val, 4),
            unit=self.unit,
            min_value=self.min_value,
            max_value=self.max_value,
            metadata=metadata or {},
        )


@MetricRegistry.register
class QualityScoreMetric(BaseBenchmarkMetric):
    """Calculates average quality score (0.0 to 10.0 scale)."""

    metric_id = "quality_score"
    name = "Quality Score"
    metric_type = "quality_score"
    unit = "pts"
    min_value = 0.0
    max_value = 10.0

    def compute(
        self,
        predictions: list[Any],
        targets: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkMetric:
        if not predictions:
            val = 0.0
        else:
            scores = [float(p) for p in predictions if isinstance(p, (int, float))]
            val = sum(scores) / len(scores) if scores else 0.0

        val = max(0.0, min(10.0, val))
        return BenchmarkMetric(
            metric_id=self.metric_id,
            name=self.name,
            metric_type=self.metric_type,
            value=round(val, 2),
            unit=self.unit,
            min_value=self.min_value,
            max_value=self.max_value,
            metadata=metadata or {},
        )


@MetricRegistry.register
class ConsistencyScoreMetric(BaseBenchmarkMetric):
    """Calculates consistency score based on standard deviation of scores (10 - std)."""

    metric_id = "consistency_score"
    name = "Consistency Score"
    metric_type = "consistency_score"
    unit = "pts"
    min_value = 0.0
    max_value = 10.0

    def compute(
        self,
        predictions: list[Any],
        targets: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkMetric:
        if not predictions:
            val = 0.0
        else:
            scores = [float(p) for p in predictions if isinstance(p, (int, float))]
            if len(scores) <= 1:
                val = 10.0
            else:
                mean = sum(scores) / len(scores)
                variance = sum((x - mean) ** 2 for x in scores) / len(scores)
                std_dev = math.sqrt(variance)
                val = max(0.0, 10.0 - std_dev)

        return BenchmarkMetric(
            metric_id=self.metric_id,
            name=self.name,
            metric_type=self.metric_type,
            value=round(val, 2),
            unit=self.unit,
            min_value=self.min_value,
            max_value=self.max_value,
            metadata=metadata or {},
        )


@MetricRegistry.register
class ConfidenceMetric(BaseBenchmarkMetric):
    """Calculates average model/reviewer confidence (0.0 to 1.0 scale)."""

    metric_id = "confidence"
    name = "Confidence"
    metric_type = "confidence"
    unit = "ratio"
    min_value = 0.0
    max_value = 1.0

    def compute(
        self,
        predictions: list[Any],
        targets: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkMetric:
        if not predictions:
            val = 0.0
        else:
            conf_list = [float(p) for p in predictions if isinstance(p, (int, float))]
            val = sum(conf_list) / len(conf_list) if conf_list else 1.0

        val = max(0.0, min(1.0, val))
        return BenchmarkMetric(
            metric_id=self.metric_id,
            name=self.name,
            metric_type=self.metric_type,
            value=round(val, 4),
            unit=self.unit,
            min_value=self.min_value,
            max_value=self.max_value,
            metadata=metadata or {},
        )


@MetricRegistry.register
class LatencyMetric(BaseBenchmarkMetric):
    """Calculates average latency in milliseconds."""

    metric_id = "latency"
    name = "Average Latency"
    metric_type = "latency"
    unit = "ms"
    min_value = 0.0
    max_value = 10000.0

    def compute(
        self,
        predictions: list[Any],
        targets: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkMetric:
        if not predictions:
            val = 0.0
        else:
            latencies = [float(p) for p in predictions if isinstance(p, (int, float))]
            val = sum(latencies) / len(latencies) if latencies else 0.0

        return BenchmarkMetric(
            metric_id=self.metric_id,
            name=self.name,
            metric_type=self.metric_type,
            value=round(val, 2),
            unit=self.unit,
            min_value=self.min_value,
            max_value=self.max_value,
            metadata=metadata or {},
        )


@MetricRegistry.register
class TokenUsageMetric(BaseBenchmarkMetric):
    """Calculates total token usage."""

    metric_id = "token_usage"
    name = "Token Usage"
    metric_type = "token_usage"
    unit = "tokens"
    min_value = 0.0
    max_value = 1000000.0

    def compute(
        self,
        predictions: list[Any],
        targets: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkMetric:
        if not predictions:
            val = 0.0
        else:
            tokens = [float(p) for p in predictions if isinstance(p, (int, float))]
            val = sum(tokens) if tokens else 0.0

        return BenchmarkMetric(
            metric_id=self.metric_id,
            name=self.name,
            metric_type=self.metric_type,
            value=round(val, 0),
            unit=self.unit,
            min_value=self.min_value,
            max_value=self.max_value,
            metadata=metadata or {},
        )


@MetricRegistry.register
class CostMetric(BaseBenchmarkMetric):
    """Calculates estimated financial cost in USD."""

    metric_id = "cost"
    name = "Estimated Cost"
    metric_type = "cost"
    unit = "USD"
    min_value = 0.0
    max_value = 100.0

    def compute(
        self,
        predictions: list[Any],
        targets: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkMetric:
        if not predictions:
            val = 0.0
        else:
            costs = [float(p) for p in predictions if isinstance(p, (int, float))]
            val = sum(costs) if costs else 0.0

        return BenchmarkMetric(
            metric_id=self.metric_id,
            name=self.name,
            metric_type=self.metric_type,
            value=round(val, 4),
            unit=self.unit,
            min_value=self.min_value,
            max_value=self.max_value,
            metadata=metadata or {},
        )


@MetricRegistry.register
class ResponseLengthMetric(BaseBenchmarkMetric):
    """Calculates average response length in characters."""

    metric_id = "response_length"
    name = "Average Response Length"
    metric_type = "response_length"
    unit = "chars"
    min_value = 0.0
    max_value = 10000.0

    def compute(
        self,
        predictions: list[Any],
        targets: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkMetric:
        if not predictions:
            val = 0.0
        else:
            lengths = [len(str(p)) for p in predictions]
            val = sum(lengths) / len(lengths)

        return BenchmarkMetric(
            metric_id=self.metric_id,
            name=self.name,
            metric_type=self.metric_type,
            value=round(val, 1),
            unit=self.unit,
            min_value=self.min_value,
            max_value=self.max_value,
            metadata=metadata or {},
        )


@MetricRegistry.register
class PromptLengthMetric(BaseBenchmarkMetric):
    """Calculates average prompt length in characters."""

    metric_id = "prompt_length"
    name = "Average Prompt Length"
    metric_type = "prompt_length"
    unit = "chars"
    min_value = 0.0
    max_value = 10000.0

    def compute(
        self,
        predictions: list[Any],
        targets: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkMetric:
        if not predictions:
            val = 0.0
        else:
            lengths = [len(str(p)) for p in predictions]
            val = sum(lengths) / len(lengths)

        return BenchmarkMetric(
            metric_id=self.metric_id,
            name=self.name,
            metric_type=self.metric_type,
            value=round(val, 1),
            unit=self.unit,
            min_value=self.min_value,
            max_value=self.max_value,
            metadata=metadata or {},
        )
