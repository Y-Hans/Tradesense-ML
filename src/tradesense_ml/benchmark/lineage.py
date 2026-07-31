"""Benchmark lineage and provenance tracking generator."""

import hashlib
import json
from datetime import datetime
from typing import Any

from tradesense_ml.domain.schemas.benchmark import BenchmarkLineage
from tradesense_ml.domain.schemas.dataset import DatasetArtifact


class BenchmarkLineageTracker:
    """Tracks provenance and complete execution lineage for benchmark evaluations."""

    @staticmethod
    def create_lineage(
        dataset_artifact: DatasetArtifact,
        config_dict: dict[str, Any],
        teacher_model: str = "teacher_llm_v1",
        student_model: str | None = None,
        prompt_version: str = "v1",
        benchmark_version: str = "v1.0.0",
        suite_version: str = "v1.0.0",
        random_seed: int = 42,
    ) -> BenchmarkLineage:
        """Construct immutable BenchmarkLineage object.

        Args:
            dataset_artifact: Target dataset evaluated.
            config_dict: Full hydra configuration dictionary.
            teacher_model: Teacher model under evaluation.
            student_model: Optional student model under evaluation.
            prompt_version: Prompt template version.
            benchmark_version: Benchmark pipeline release version.
            suite_version: Benchmark suite semantic version.
            random_seed: Seed for deterministic execution.

        Returns:
            BenchmarkLineage instance with SHA-256 configuration hash.
        """
        # Create deterministic SHA-256 configuration hash
        config_bytes = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
        config_hash = hashlib.sha256(config_bytes).hexdigest()

        metric_versions = {
            "accuracy": "v1.0.0",
            "pass_rate": "v1.0.0",
            "quality_score": "v1.0.0",
            "consistency_score": "v1.0.0",
            "confidence": "v1.0.0",
            "latency": "v1.0.0",
            "token_usage": "v1.0.0",
            "cost": "v1.0.0",
            "response_length": "v1.0.0",
            "prompt_length": "v1.0.0",
        }

        return BenchmarkLineage(
            benchmark_version=benchmark_version,
            dataset_artifact_id=dataset_artifact.artifact_id,
            dataset_version=dataset_artifact.dataset_metadata.version,
            teacher_model=teacher_model,
            student_model=student_model,
            prompt_version=prompt_version,
            configuration_hash=config_hash,
            metric_versions=metric_versions,
            benchmark_suite_version=suite_version,
            execution_timestamp=datetime.utcnow(),
            random_seed=random_seed,
        )
