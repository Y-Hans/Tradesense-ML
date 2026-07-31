"""Dedicated DatasetBuilderPipeline orchestrating dataset transformation, splitting, validation, artifact creation, and export."""

import time
from pathlib import Path
from typing import Any

from tradesense_ml.dataset.exporters import DatasetExporterManager
from tradesense_ml.dataset.filtering import DatasetFilter
from tradesense_ml.dataset.lineage import DatasetLineageTracker
from tradesense_ml.dataset.manifest import DatasetManifestGenerator
from tradesense_ml.dataset.models import DatasetArtifact, DatasetMetadata
from tradesense_ml.dataset.splitting import DatasetSplitter
from tradesense_ml.dataset.statistics import DatasetStatisticsGenerator
from tradesense_ml.dataset.transformation import DatasetTransformer
from tradesense_ml.dataset.validation import DatasetValidator
from tradesense_ml.logging.logger import get_logger
from tradesense_ml.pipelines.base import BasePipeline

logger = get_logger()


class DatasetBuilderPipeline(BasePipeline[list[Any], DatasetArtifact]):
    """Orchestrator pipeline transforming reviewed coaching examples into reproducible, versioned DatasetArtifact releases."""

    def __init__(
        self,
        filter_engine: DatasetFilter | None = None,
        transformer: DatasetTransformer | None = None,
        splitter: DatasetSplitter | None = None,
        validator: DatasetValidator | None = None,
        dataset_id: str = "tradesense_sft_v1",
        version: str = "v1.0.0",
    ) -> None:
        super().__init__(pipeline_name="dataset_builder_pipeline")
        self.filter_engine = filter_engine or DatasetFilter()
        self.transformer = transformer or DatasetTransformer()
        self.splitter = splitter or DatasetSplitter()
        self.validator = validator or DatasetValidator()
        self.dataset_id = dataset_id
        self.version = version

    def run(self, input_data: list[Any], **kwargs: Any) -> DatasetArtifact:
        """Run dataset builder pipeline end-to-end.

        Flow:
        DatasetBuilderPipeline -> DatasetArtifact -> Exporters

        Args:
            input_data: List of reviewed examples, request-response-decision tuples, or raw dicts.
            **kwargs: Overrides for dataset_id, version, dataset_format, split_ratios, seed, output_dir, export_formats.

        Returns:
            DatasetArtifact object encapsulating the entire dataset release.
        """
        start_time = time.perf_counter()

        # Extract runtime configuration options
        ds_id = str(kwargs.get("dataset_id", self.dataset_id))
        ds_version = str(kwargs.get("dataset_version", kwargs.get("version", self.version)))
        ds_format = str(kwargs.get("dataset_format", "sft_instruction"))
        output_dir = str(kwargs.get("output_dir", "datasets"))
        export_formats = kwargs.get("export_formats", ["jsonl", "json", "parquet"])
        seed = int(kwargs.get("seed", 42))
        split_ratios = kwargs.get("split_ratios", {"train": 0.8, "validation": 0.1, "test": 0.1})

        logger.info(
            f"Starting DatasetBuilderPipeline for '{ds_id}:{ds_version}' (format={ds_format}, input items={len(input_data)})"
        )

        # 1. Filter approved and valid examples
        filtering_result = self.filter_engine.filter_batch(input_data)
        approved_items = filtering_result.kept_examples

        if not approved_items:
            raise ValueError(
                f"DatasetBuilderPipeline aborted: 0 examples passed filtering criteria out of {len(input_data)} evaluated."
            )

        # 2. Transform into canonical DatasetExample objects formatted according to DatasetFormat strategy
        transformed_examples = self.transformer.transform_batch(
            items=approved_items,
            target_format=ds_format,
            version=ds_version,
        )

        # 3. Partition deterministically into train/val/test splits
        splitter = DatasetSplitter(ratios=split_ratios, seed=seed)
        split_result = splitter.split(transformed_examples)

        # 4. Validate dataset integrity and split isolation
        val_report = self.validator.validate_dataset(
            examples=transformed_examples,
            split_dict=split_result.splits,
            require_review_info=True,
            require_lineage=True,
        )

        if not val_report.is_valid:
            error_msg = f"Dataset validation failed with errors: {'; '.join(val_report.errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 5. Generate aggregate statistics
        stats = DatasetStatisticsGenerator.generate(
            dataset_id=ds_id,
            examples=transformed_examples,
            total_evaluated=filtering_result.total_evaluated,
            rejected_count=filtering_result.rejected_count,
            split_sizes=split_result.split_sizes,
            version_info={"pipeline": self.pipeline_name, "version": ds_version},
        )

        # 6. Generate provenance lineage
        config_dict = {
            "dataset_id": ds_id,
            "version": ds_version,
            "format": ds_format,
            "seed": seed,
            "split_ratios": split_ratios,
            "filter_config": self.filter_engine.config.model_dump(),
        }

        lineage = DatasetLineageTracker.create_lineage(
            dataset_id=ds_id,
            dataset_version=ds_version,
            config_dict=config_dict,
            examples=transformed_examples,
            random_seed=seed,
        )

        metadata = DatasetMetadata(
            name=ds_id,
            description=f"TradeSense coaching dataset release '{ds_id}:{ds_version}' ({ds_format})",
            version=ds_version,
            extra={"seed": seed, "format": ds_format},
        )

        # 7. Create initial DatasetArtifact container
        provisional_manifest = DatasetManifestGenerator.generate_manifest(
            dataset_id=ds_id,
            version=ds_version,
            dataset_format=ds_format,
            stats=stats,
            lineage=lineage,
            export_file_paths=[],
            config_version="v1.0.0",
        )

        artifact = DatasetArtifact(
            artifact_id=ds_id,
            dataset_metadata=metadata,
            lineage=lineage,
            statistics=stats,
            manifest=provisional_manifest,
            splits=split_result.splits,
            export_files=[],
        )

        # 8. Flow DatasetArtifact into Exporters
        file_descriptors = DatasetExporterManager.export_artifact(
            artifact=artifact,
            output_dir=output_dir,
            formats=export_formats,
        )

        # 9. Update final manifest with export checksums and save to disk
        final_manifest = DatasetManifestGenerator.generate_manifest(
            dataset_id=ds_id,
            version=ds_version,
            dataset_format=ds_format,
            stats=stats,
            lineage=lineage,
            export_file_paths=file_descriptors,
            config_version="v1.0.0",
        )

        manifest_dir = Path(output_dir) / ds_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_dir / "manifest.json"
        manifest_file.write_text(final_manifest.model_dump_json(indent=2), encoding="utf-8")

        # Construct final immutable DatasetArtifact containing export file metadata and final manifest
        final_artifact = DatasetArtifact(
            artifact_id=ds_id,
            dataset_metadata=metadata,
            lineage=lineage,
            statistics=stats,
            manifest=final_manifest,
            splits=split_result.splits,
            export_files=file_descriptors,
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            f"Successfully built DatasetArtifact for '{ds_id}:{ds_version}' in {total_latency_ms:.2f}ms. "
            f"Manifest saved to: {manifest_file}"
        )

        return final_artifact
