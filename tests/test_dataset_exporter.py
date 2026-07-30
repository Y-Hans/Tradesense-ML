"""Unit tests for DatasetExporter supporting JSONL, JSON, and Parquet formats."""

from pathlib import Path

from tradesense_ml.domain.schemas.synthetic import SyntheticGeneratorConfig
from tradesense_ml.pipelines.generation.pipeline import ConcreteSyntheticGenerationPipeline
from tradesense_ml.storage.dataset_exporter import DatasetExporter


def test_exporter_jsonl_json_parquet(tmp_path: Path) -> None:
    """Test exporting and reloading synthetic datasets across JSONL, JSON, and Parquet."""
    cfg = SyntheticGeneratorConfig(num_samples=5, seed=42)
    pipeline = ConcreteSyntheticGenerationPipeline()
    requests, lineage = pipeline.generate_dataset(cfg)

    exporter = DatasetExporter()

    for fmt in ["jsonl", "json", "parquet"]:
        out_file = tmp_path / f"test_dataset.{fmt}"
        exported_path = exporter.export(
            samples=requests,
            lineage=lineage,
            output_path=out_file,
            format_type=fmt,
            validate_first=True,
        )

        assert exported_path.exists()
        meta_file = Path(f"{exported_path}.meta.json")
        assert meta_file.exists()

        reloaded = DatasetExporter.load_dataset(exported_path)
        assert len(reloaded) == 5
