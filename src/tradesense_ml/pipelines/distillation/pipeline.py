"""Distillation Pipeline Orchestrator interface.

Teacher -> Synthetic Examples -> Validation -> Review -> Training Dataset -> Fine-Tuning -> Evaluation -> Model Registry
"""

from abc import ABC, abstractmethod
from typing import Any

from tradesense_ml.domain.schemas.examples import TrainingExample
from tradesense_ml.domain.schemas.lineage import DatasetVersionMetadata
from tradesense_ml.pipelines.base import BasePipeline


class DistillationPipeline(BasePipeline[DatasetVersionMetadata, list[TrainingExample]], ABC):
    """Abstract interface for knowledge distillation dataset creation."""

    def __init__(self) -> None:
        super().__init__(pipeline_name="distillation_pipeline")

    @abstractmethod
    def run(self, input_data: DatasetVersionMetadata, **kwargs: Any) -> list[TrainingExample]:
        """Execute teacher inference, consensus filtering, and dataset packaging."""
        pass
