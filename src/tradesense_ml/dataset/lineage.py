"""Dataset provenance lineage tracker and configuration hashing module."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from tradesense_ml.domain.schemas.dataset import DatasetExample, DatasetLineage
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class DatasetLineageTracker:
    """Provenance tracker assembling reproducible lineage metadata for dataset builds."""

    @classmethod
    def create_lineage(
        cls,
        dataset_id: str,
        dataset_version: str,
        config_dict: dict[str, Any],
        examples: list[DatasetExample],
        random_seed: int = 42,
        synthetic_generator_version: str = "v1.0.0",
        teacher_inference_version: str = "v1.0.0",
        review_version: str = "v1.0.0",
        prompt_version: str = "v1",
        review_criteria_version: str = "v1.0.0",
    ) -> DatasetLineage:
        """Assemble DatasetLineage instance tracking full build provenance."""
        config_bytes = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
        config_hash = hashlib.sha256(config_bytes).hexdigest()

        source_ids = [ex.example_id for ex in examples]

        lineage = DatasetLineage(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            generation_timestamp=datetime.now(UTC),
            random_seed=random_seed,
            synthetic_generator_version=synthetic_generator_version,
            teacher_inference_version=teacher_inference_version,
            review_version=review_version,
            configuration_hash=config_hash,
            prompt_version=prompt_version,
            review_criteria_version=review_criteria_version,
            source_example_ids=source_ids,
        )

        logger.info(
            f"Created DatasetLineage for '{dataset_id}:{dataset_version}' (config hash={config_hash[:12]}..., {len(source_ids)} source examples)"
        )
        return lineage
