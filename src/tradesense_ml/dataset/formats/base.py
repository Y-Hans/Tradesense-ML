"""Base class interface for dataset formatters."""

from abc import ABC, abstractmethod

from tradesense_ml.domain.schemas.dataset import DatasetExample


class BaseDatasetFormat(ABC):
    """Abstract base class defining the DatasetFormat interface."""

    def __init__(self, format_name: str) -> None:
        self.format_name = format_name

    @abstractmethod
    def format_example(self, example: DatasetExample) -> DatasetExample:
        """Format a canonical DatasetExample according to target dataset schema/objective rules.

        Args:
            example: Raw canonical DatasetExample record.

        Returns:
            New formatted DatasetExample record updated with target schema fields.
        """
        pass
