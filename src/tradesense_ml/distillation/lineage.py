"""Lineage and provenance tracker for distillation runs."""

import hashlib
import json
from datetime import datetime
from typing import Any

from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.dataset import DatasetArtifact
from tradesense_ml.domain.schemas.distillation import DistillationLineage


class DistillationLineageTracker:
    """Tracker generating SHA-256 configuration hashes and complete lineage records for distillation."""

    @staticmethod
    def create_lineage(
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        config_dict: dict[str, Any] | None = None,
        distillation_strategy: str = "SFTStrategy",
        selection_strategy: str = "ThresholdSelection",
        sampling_strategy: str = "UniformSampling",
        curriculum_strategy: str = "StandardCurriculumStrategy",
        teacher_model: str = "teacher_llm_v1",
        prompt_version: str = "v1",
        random_seed: int = 42,
        repository_version: str = "v1.0.0",
    ) -> DistillationLineage:
        cfg = config_dict or {}
        # Deterministic SHA-256 hash of configuration
        serialized_config = json.dumps(cfg, sort_keys=True, default=str)
        config_hash = hashlib.sha256(serialized_config.encode("utf-8")).hexdigest()

        bm_id = benchmark_artifact.artifact_id if benchmark_artifact else "none"

        return DistillationLineage(
            dataset_artifact_id=dataset_artifact.artifact_id,
            benchmark_artifact_id=bm_id,
            teacher_model=teacher_model,
            prompt_version=prompt_version,
            selection_strategy=selection_strategy,
            sampling_strategy=sampling_strategy,
            curriculum_strategy=curriculum_strategy,
            distillation_strategy=distillation_strategy,
            configuration_hash=config_hash,
            random_seed=random_seed,
            execution_timestamp=datetime.utcnow(),
            repository_version=repository_version,
        )
