"""Dataset manifest generation engine with file checksum tracking."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradesense_ml.domain.schemas.dataset import (
    DatasetLineage,
    DatasetManifest,
    DatasetStatistics,
)
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class DatasetManifestGenerator:
    """Generator constructing immutable dataset manifests with SHA256 checksums."""

    @classmethod
    def compute_file_sha256(cls, file_path: Path | str) -> tuple[str, int]:
        """Compute SHA256 hex digest and byte size of a file."""
        path = Path(file_path)
        if not path.exists():
            return "", 0

        sha = hashlib.sha256()
        size_bytes = 0
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
                size_bytes += len(chunk)

        return sha.hexdigest(), size_bytes

    @classmethod
    def generate_manifest(
        cls,
        dataset_id: str,
        version: str,
        dataset_format: str,
        stats: DatasetStatistics,
        lineage: DatasetLineage,
        export_file_paths: list[dict[str, Any]],
        config_version: str = "v1.0.0",
    ) -> DatasetManifest:
        """Construct DatasetManifest detailing export files, split sizes, and SHA256 checksums."""
        export_files: list[dict[str, Any]] = []

        for item in export_file_paths:
            fpath = item.get("path")
            split_name = item.get("split", "full")
            count = item.get("count", 0)

            if fpath:
                path_obj = Path(fpath)
                checksum, size_bytes = cls.compute_file_sha256(path_obj)
                export_files.append(
                    {
                        "file_name": path_obj.name,
                        "file_path": str(path_obj),
                        "split": split_name,
                        "example_count": count,
                        "size_bytes": size_bytes,
                        "sha256_checksum": checksum,
                    }
                )

        manifest_data = {
            "dataset_id": dataset_id,
            "version": version,
            "dataset_format": dataset_format,
            "split_sizes": stats.split_sizes,
            "statistics_summary": stats.model_dump(mode="json"),
            "configuration_version": config_version,
            "lineage": lineage.model_dump(mode="json"),
            "export_files": export_files,
        }

        manifest_bytes = json.dumps(manifest_data, sort_keys=True, default=str).encode("utf-8")
        manifest_checksum = hashlib.sha256(manifest_bytes).hexdigest()

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            version=version,
            creation_timestamp=datetime.now(UTC),
            dataset_format=dataset_format,
            split_sizes=stats.split_sizes,
            statistics_summary=stats.model_dump(mode="json"),
            configuration_version=config_version,
            lineage=lineage.model_dump(mode="json"),
            export_files=export_files,
            checksum=manifest_checksum,
        )

        logger.info(
            f"Generated DatasetManifest for '{dataset_id}:{version}' (checksum={manifest_checksum[:12]}...)"
        )
        return manifest
