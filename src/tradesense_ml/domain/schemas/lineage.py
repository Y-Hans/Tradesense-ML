"""Dataset metadata and complete provenance lineage schema."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.review import ReviewStage


class DatasetSplit(str, Enum):
    """Dataset split classification."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BENCHMARK = "benchmark"


class DatasetVersionMetadata(BaseModel):
    """Comprehensive dataset versioning and provenance lineage model.

    Tracks full lineage (teacher model, prompt version, rubric version,
    generator version, review version, dataset version, timestamps, source hash).
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(
        ..., description="Unique dataset identifier, e.g. tradesense_coaching_v1"
    )
    dataset_version: str = Field(..., description="Semantic dataset version, e.g. 1.2.0")
    parent_dataset_id: str | None = Field(
        default=None, description="ID of parent dataset if derived/filtered"
    )

    # Lineage & Provenance Tracking
    teacher_model: str = Field(..., description="Teacher model ID used for generation/evaluation")
    prompt_version: str = Field(..., description="Version of system prompt template used")
    rubric_version: str = Field(..., description="Version of evaluation rubric applied")
    generator_version: str = Field(..., description="Version of synthetic generator script")
    review_version: str = Field(..., description="Version of review pipeline executed")

    # Timestamps & Hashing
    generation_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Creation UTC timestamp"
    )
    source_hash: str = Field(..., description="SHA-256 hash of raw input source data")

    # Status & Quality Metadata
    review_status: ReviewStage = Field(
        default=ReviewStage.AUTOMATED_VALIDATION, description="Current review lifecycle stage"
    )
    quality_score: float | None = Field(
        default=None, ge=0.0, le=10.0, description="Average quality score across dataset"
    )
    difficulty: str = Field(default="medium", description="Difficulty level rating")
    tags: list[str] = Field(default_factory=list, description="Categorical tags")
    split: DatasetSplit = Field(default=DatasetSplit.TRAIN, description="Data split")
    sample_count: int = Field(default=0, ge=0, description="Total examples in dataset")

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary custom lineage metadata"
    )
