"""Dataset exporter module supporting JSONL, JSON, and Parquet output formats."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tradesense_ml.domain.schemas.coaching import CoachRequest
from tradesense_ml.domain.schemas.lineage import DatasetVersionMetadata
from tradesense_ml.pipelines.validation.synthetic_validator import SyntheticDatasetValidator


class DatasetExporter:
    """Exporter for saving synthetic datasets in JSONL, JSON, and Parquet formats."""

    def __init__(self, validator: SyntheticDatasetValidator | None = None) -> None:
        """Initialize dataset exporter with optional validator instance."""
        self.validator = validator or SyntheticDatasetValidator()

    def export(
        self,
        samples: list[CoachRequest],
        lineage: DatasetVersionMetadata,
        output_path: str | Path,
        format_type: str = "jsonl",
        validate_first: bool = True,
    ) -> Path:
        """Export dataset samples and lineage provenance metadata to specified output path."""
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if validate_first:
            all_valid, results = self.validator.validate_batch(samples)
            if not all_valid:
                failed_errors = [
                    f"Sample #{i}: {', '.join(r.errors)}"
                    for i, r in enumerate(results)
                    if not r.is_valid
                ]
                raise ValueError(
                    f"Dataset validation failed for {len(failed_errors)} samples:\n"
                    + "\n".join(failed_errors[:5])
                )

        # Convert Pydantic models to dict lists
        dict_samples: list[dict[str, Any]] = [s.model_dump(mode="json") for s in samples]

        fmt = format_type.lower()
        if fmt == "jsonl" or target_path.suffix == ".jsonl":
            with open(target_path, "w", encoding="utf-8") as f:
                for record in dict_samples:
                    f.write(json.dumps(record, default=str) + "\n")

        elif fmt == "json" or target_path.suffix == ".json":
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(dict_samples, f, indent=2, default=str)

        elif fmt == "parquet" or target_path.suffix == ".parquet":
            # Flatten or store dict records in parquet dataframe
            df = pd.DataFrame(dict_samples)
            df.to_parquet(target_path, index=False)

        else:
            raise ValueError(
                f"Unsupported export format '{format_type}'. Must be jsonl, json, or parquet."
            )

        # Write sidecar lineage file
        lineage_path = target_path.with_suffix(target_path.suffix + ".meta.json")
        with open(lineage_path, "w", encoding="utf-8") as f:
            f.write(lineage.model_dump_json(indent=2))

        return target_path

    @staticmethod
    def load_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
        """Load dataset records from JSONL, JSON, or Parquet path."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        if path.suffix == ".jsonl":
            records: list[dict[str, Any]] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records

        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]

        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
            return df.to_dict(orient="records")  # type: ignore[no-any-return]

        else:
            raise ValueError(f"Unsupported dataset format suffix: '{path.suffix}'")
