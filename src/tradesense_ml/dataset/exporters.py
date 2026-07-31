"""Dataset exporters writing DatasetExample objects into JSON, JSONL, and Parquet formats."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tradesense_ml.domain.schemas.dataset import DatasetArtifact, DatasetExample
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class BaseDatasetExporter(ABC):
    """Abstract base class for format-specific dataset exporters."""

    def __init__(self, format_name: str) -> None:
        self.format_name = format_name

    @abstractmethod
    def export(
        self,
        examples: list[DatasetExample],
        output_path: Path | str,
    ) -> Path:
        """Export list of DatasetExample records to output file path."""
        pass


class JSONLExporter(BaseDatasetExporter):
    """JSONL format exporter (one JSON object per line)."""

    def __init__(self) -> None:
        super().__init__(format_name="jsonl")

    def export(
        self,
        examples: list[DatasetExample],
        output_path: Path | str,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [e.model_dump_json() for e in examples]
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Exported {len(examples)} records to JSONL file: {path}")
        return path


class JSONExporter(BaseDatasetExporter):
    """JSON array format exporter."""

    def __init__(self) -> None:
        super().__init__(format_name="json")

    def export(
        self,
        examples: list[DatasetExample],
        output_path: Path | str,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        items = [e.model_dump(mode="json") for e in examples]
        path.write_text(json.dumps(items, indent=2, default=str), encoding="utf-8")
        logger.info(f"Exported {len(examples)} records to JSON file: {path}")
        return path


class ParquetExporter(BaseDatasetExporter):
    """Parquet columnar format exporter with PyArrow/Pandas fallback."""

    def __init__(self) -> None:
        super().__init__(format_name="parquet")

    def export(
        self,
        examples: list[DatasetExample],
        output_path: Path | str,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        items = [e.model_dump(mode="json") for e in examples]

        try:
            import pandas as pd

            df = pd.DataFrame(items)
            # Convert dict/list columns to string JSON for Parquet compatibility
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    df[col] = df[col].apply(
                        lambda x: (
                            json.dumps(x, default=str) if isinstance(x, (dict, list)) else str(x)
                        )
                    )
            df.to_parquet(path, index=False)
            logger.info(f"Exported {len(examples)} records to Parquet file: {path}")
            return path
        except ImportError:
            # Fallback if pandas/parquet library unavailable
            fallback_path = path.with_suffix(".json")
            logger.warning(
                f"Pandas/PyArrow unavailable for Parquet export; falling back to JSON: {fallback_path}"
            )
            json_exp = JSONExporter()
            return json_exp.export(examples, fallback_path)


class DatasetExporterManager:
    """Manager orchestrating dataset split exports across multiple requested formats."""

    @classmethod
    def get_exporter(cls, fmt: str) -> BaseDatasetExporter:
        """Factory creating format-specific exporter instance."""
        if fmt == "json":
            return JSONExporter()
        elif fmt == "parquet":
            return ParquetExporter()
        else:
            return JSONLExporter()

    @classmethod
    def export_splits(
        cls,
        splits: dict[str, list[DatasetExample]],
        output_dir: Path | str,
        dataset_id: str,
        formats: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Export dataset splits to output directory in requested file formats.

        Args:
            splits: Map of split name (train, validation, test) to list of DatasetExample objects.
            output_dir: Base output directory path.
            dataset_id: Dataset identifier.
            formats: List of formats to export (e.g. ['jsonl', 'json', 'parquet']).

        Returns:
            List of exported file descriptors containing path, split name, count.
        """
        out_dir = Path(output_dir) / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)

        requested_formats = formats or ["jsonl", "json", "parquet"]
        file_descriptors: list[dict[str, Any]] = []

        for fmt in requested_formats:
            fmt_lower = fmt.lower().strip()
            exporter = cls.get_exporter(fmt_lower)

            for split_name, examples in splits.items():
                filename = f"{dataset_id}_{split_name}.{fmt_lower}"
                filepath = out_dir / filename

                exported_path = exporter.export(examples, filepath)
                file_descriptors.append(
                    {
                        "path": exported_path,
                        "split": split_name,
                        "count": len(examples),
                        "format": fmt_lower,
                    }
                )

        return file_descriptors

    @classmethod
    def export_artifact(
        cls,
        artifact: DatasetArtifact,
        output_dir: Path | str,
        formats: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Export a DatasetArtifact's split datasets to output directory.

        Args:
            artifact: The complete DatasetArtifact payload.
            output_dir: Base output directory.
            formats: List of formats to export.

        Returns:
            List of exported file descriptors containing file path, split, and count.
        """
        return cls.export_splits(
            splits=artifact.splits,
            output_dir=output_dir,
            dataset_id=artifact.artifact_id,
            formats=formats,
        )
