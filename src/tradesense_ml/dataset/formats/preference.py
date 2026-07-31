"""Preference optimization and reward model dataset format scaffolds."""

from tradesense_ml.dataset.formats.base import BaseDatasetFormat
from tradesense_ml.domain.schemas.dataset import DatasetExample


class DPOFormat(BaseDatasetFormat):
    """Direct Preference Optimization (DPO) dataset format scaffold."""

    def __init__(self) -> None:
        super().__init__(format_name="dpo")

    def format_example(self, example: DatasetExample) -> DatasetExample:
        meta = dict(example.metadata)
        meta["preference_pair"] = {
            "chosen": example.output,
            "rejected": example.metadata.get("rejected_output", "N/A"),
        }
        return example.model_copy(
            update={
                "format_type": self.format_name,
                "metadata": meta,
            }
        )


class ORPOFormat(BaseDatasetFormat):
    """Odds Ratio Preference Optimization (ORPO) dataset format scaffold."""

    def __init__(self) -> None:
        super().__init__(format_name="orpo")

    def format_example(self, example: DatasetExample) -> DatasetExample:
        meta = dict(example.metadata)
        meta["orpo_pair"] = {
            "chosen": example.output,
            "rejected": example.metadata.get("rejected_output", "N/A"),
        }
        return example.model_copy(
            update={
                "format_type": self.format_name,
                "metadata": meta,
            }
        )


class KTOFormat(BaseDatasetFormat):
    """Kahneman-Tversky Optimization (KTO) dataset format scaffold."""

    def __init__(self) -> None:
        super().__init__(format_name="kto")

    def format_example(self, example: DatasetExample) -> DatasetExample:
        meta = dict(example.metadata)
        meta["kto_binary_label"] = example.review_info.get("quality_score", 0.0) >= 7.0
        return example.model_copy(
            update={
                "format_type": self.format_name,
                "metadata": meta,
            }
        )


class RewardModelFormat(BaseDatasetFormat):
    """Reward Model training dataset format scaffold."""

    def __init__(self) -> None:
        super().__init__(format_name="reward_model")

    def format_example(self, example: DatasetExample) -> DatasetExample:
        meta = dict(example.metadata)
        meta["reward_score"] = example.review_info.get("quality_score", 0.0)
        return example.model_copy(
            update={
                "format_type": self.format_name,
                "metadata": meta,
            }
        )
