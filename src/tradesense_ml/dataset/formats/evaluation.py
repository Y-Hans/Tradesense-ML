"""Evaluation dataset format for benchmarking and ground truth comparison."""

from tradesense_ml.dataset.formats.base import BaseDatasetFormat
from tradesense_ml.domain.schemas.dataset import DatasetExample


class EvaluationFormat(BaseDatasetFormat):
    """Evaluation benchmark format for downstream coach evaluation."""

    def __init__(self) -> None:
        super().__init__(format_name="evaluation")

    def format_example(self, example: DatasetExample) -> DatasetExample:
        eval_metadata = dict(example.metadata)
        eval_metadata["is_ground_truth"] = True
        eval_metadata["expected_quality_score"] = example.review_info.get("quality_score", 10.0)

        prompt_text = (
            f"[EVALUATION PROMPT]\nInstruction: {example.instruction}\nInput:\n{example.input}"
        )

        return example.model_copy(
            update={
                "format_type": self.format_name,
                "prompt": prompt_text,
                "metadata": eval_metadata,
            }
        )
