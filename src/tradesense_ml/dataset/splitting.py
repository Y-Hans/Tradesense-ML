"""Deterministic dataset splitting module with configurable ratios and seed reproducibility."""

import random

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.dataset import DatasetExample
from tradesense_ml.domain.schemas.lineage import DatasetSplit
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class SplitResult(BaseModel):
    """Container holding split subsets."""

    model_config = ConfigDict(frozen=True)

    splits: dict[str, list[DatasetExample]] = Field(
        ...,
        description="Map of split name (train, validation, test) to list of DatasetExample objects",
    )
    split_sizes: dict[str, int] = Field(..., description="Count of examples per split")
    seed: int = Field(..., description="Random seed used for partitioning")


class DatasetSplitter:
    """Deterministic splitter partitioning dataset examples into train, validation, and test sets."""

    DEFAULT_RATIOS: dict[str, float] = {
        DatasetSplit.TRAIN.value: 0.8,
        DatasetSplit.VALIDATION.value: 0.1,
        DatasetSplit.TEST.value: 0.1,
    }

    def __init__(
        self,
        ratios: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        self.ratios = ratios or dict(self.DEFAULT_RATIOS)
        self.seed = seed

        # Normalize ratios if sum != 1.0
        total = sum(self.ratios.values())
        if total > 0 and abs(total - 1.0) > 1e-5:
            self.ratios = {k: v / total for k, v in self.ratios.items()}

    def split(
        self,
        examples: list[DatasetExample],
        stratify_by: str | None = None,
    ) -> SplitResult:
        """Partition list of DatasetExample objects deterministically into split subsets.

        Args:
            examples: List of DatasetExample records to partition.
            stratify_by: Optional metadata field for stratification (extension hook).

        Returns:
            SplitResult containing split dictionaries and sizes.
        """
        if not examples:
            return SplitResult(
                splits={"train": [], "validation": [], "test": []},
                split_sizes={"train": 0, "validation": 0, "test": 0},
                seed=self.seed,
            )

        # Shuffle deterministically using explicit Random instance
        rng = random.Random(self.seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_ratio = self.ratios.get(DatasetSplit.TRAIN.value, 0.8)
        val_ratio = self.ratios.get(DatasetSplit.VALIDATION.value, 0.1)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Ensure at least 1 sample per split if n >= 3
        if n >= 3 and train_end == 0:
            train_end = 1
        if n >= 3 and val_end <= train_end:
            val_end = train_end + 1

        train_set = shuffled[:train_end]
        val_set = shuffled[train_end:val_end]
        test_set = shuffled[val_end:]

        splits = {
            DatasetSplit.TRAIN.value: train_set,
            DatasetSplit.VALIDATION.value: val_set,
            DatasetSplit.TEST.value: test_set,
        }

        sizes = {k: len(v) for k, v in splits.items()}

        logger.info(
            f"Split dataset ({n} items, seed={self.seed}): train={sizes['train']}, val={sizes['validation']}, test={sizes['test']}"
        )

        return SplitResult(splits=splits, split_sizes=sizes, seed=self.seed)

    def stratify_hook(
        self, examples: list[DatasetExample], key: str
    ) -> dict[str, list[DatasetExample]]:
        """Extension hook for categorical stratification partitioning."""
        groups: dict[str, list[DatasetExample]] = {}
        for ex in examples:
            group_key = str(ex.metadata.get(key, ex.review_info.get(key, "default")))
            groups.setdefault(group_key, []).append(ex)
        return groups
