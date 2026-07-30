"""Ingestion pipeline interface."""

from abc import ABC, abstractmethod
from typing import Any

from tradesense_ml.domain.schemas.trade import Trade
from tradesense_ml.pipelines.base import BasePipeline


class BaseIngestionPipeline(BasePipeline[Any, list[Trade]], ABC):
    """Abstract pipeline for ingesting trade records from external platforms/APIs."""

    def __init__(self) -> None:
        super().__init__(pipeline_name="ingestion_pipeline")

    @abstractmethod
    def run(self, input_data: Any, **kwargs: Any) -> list[Trade]:
        """Import raw data into clean Trade objects."""
        pass
