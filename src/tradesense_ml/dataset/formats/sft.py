"""Supervised Fine-Tuning (SFT) dataset formats."""

from tradesense_ml.dataset.formats.base import BaseDatasetFormat
from tradesense_ml.domain.schemas.dataset import DatasetExample


class SFTInstructionFormat(BaseDatasetFormat):
    """SFT Instruction format (instruction, input, output, prompt)."""

    def __init__(self) -> None:
        super().__init__(format_name="sft_instruction")

    def format_example(self, example: DatasetExample) -> DatasetExample:
        prompt_text = f"{example.instruction}\n\n### User Request:\n{example.input}\n\n### Coaching Response:\n"
        return example.model_copy(
            update={
                "format_type": self.format_name,
                "prompt": prompt_text,
            }
        )


class SFTChatFormat(BaseDatasetFormat):
    """SFT Chat format (messages list containing system, user, assistant roles)."""

    def __init__(self) -> None:
        super().__init__(format_name="sft_chat")

    def format_example(self, example: DatasetExample) -> DatasetExample:
        messages = [
            {"role": "system", "content": example.instruction},
            {"role": "user", "content": example.input},
            {"role": "assistant", "content": example.output},
        ]
        prompt_text = (
            f"<|im_start|>system\n{example.instruction}<|im_end|>\n"
            f"<|im_start|>user\n{example.input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return example.model_copy(
            update={
                "format_type": self.format_name,
                "messages": messages,
                "prompt": prompt_text,
            }
        )
