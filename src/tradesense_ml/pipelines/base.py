"""Base pipeline interface for AI Lifecycle execution."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class BasePipeline(ABC, Generic[InT, OutT]):
    """Abstract interface for all TradeSense AI lifecycle pipelines."""

    def __init__(self, pipeline_name: str) -> None:
        self.pipeline_name = pipeline_name

    @abstractmethod
    def run(self, input_data: InT, **kwargs: Any) -> OutT:
        """Execute pipeline stage."""
        pass
