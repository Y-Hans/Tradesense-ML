"""Deterministic Mock Training Backend for testing and non-GPU simulation."""

import json
import math
from pathlib import Path
from typing import Any

from tradesense_ml.domain.schemas.distillation import DistillationArtifact
from tradesense_ml.domain.schemas.finetuning import (
    ModelCheckpoint,
    TrainingBackendResult,
    TrainingConfiguration,
    TrainingMetrics,
)
from tradesense_ml.finetuning.backends.base import TrainingBackend


class MockBackend(TrainingBackend):
    """Deterministic mock training backend that simulates LLM fine-tuning without GPU dependencies."""

    @property
    def name(self) -> str:
        return "mock"

    def train(
        self,
        distillation_artifact: DistillationArtifact,
        config: TrainingConfiguration,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
    ) -> TrainingBackendResult:
        out_path = Path(output_dir)
        weights_dir = out_path / "weights"
        checkpoints_dir = out_path / "checkpoints"
        weights_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        params = config.model_config_params
        total_epochs = params.num_epochs
        samples_count = (
            len(distillation_artifact.dataset.sft_examples)
            or len(distillation_artifact.dataset.preference_pairs)
            or 100
        )
        effective_batch_size = (
            params.per_device_train_batch_size * params.gradient_accumulation_steps
        )
        steps_per_epoch = max(1, math.ceil(samples_count / max(1, effective_batch_size)))
        total_steps = steps_per_epoch * total_epochs

        start_step = 0
        if resume_from_checkpoint and "step-" in resume_from_checkpoint:
            try:
                start_step = int(
                    resume_from_checkpoint.split("step-")[-1].split("/")[0].split("\\")[0]
                )
            except ValueError:
                start_step = 0

        loss_history: list[dict[str, Any]] = []
        eval_loss_history: list[dict[str, Any]] = []
        lr_history: list[float] = []
        checkpoints: list[ModelCheckpoint] = []

        initial_loss = 2.80
        target_loss = 0.45

        for step in range(start_step, total_steps + 1):
            if step == 0:
                continue

            progress = step / total_steps
            # Decay curve formula
            current_loss = target_loss + (initial_loss - target_loss) * math.exp(-3.0 * progress)
            current_lr = params.learning_rate * (1.0 - progress * 0.8)

            loss_history.append(
                {"step": step, "loss": round(current_loss, 4), "lr": round(current_lr, 6)}
            )
            lr_history.append(round(current_lr, 6))

            # Evaluation steps
            if step % params.eval_interval_steps == 0 or step == total_steps:
                eval_loss = current_loss * 1.05 + 0.02
                eval_loss_history.append({"step": step, "eval_loss": round(eval_loss, 4)})

            # Checkpoint steps
            if step % params.checkpoint_interval_steps == 0 or step == total_steps:
                ckpt_dir = checkpoints_dir / f"step-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_meta = {
                    "checkpoint_id": f"step-{step}",
                    "step": step,
                    "epoch": round(step / steps_per_epoch, 2),
                    "loss": round(current_loss, 4),
                    "model_name": params.base_model_name_or_path,
                }
                with open(ckpt_dir / "checkpoint_info.json", "w", encoding="utf-8") as f:
                    json.dump(ckpt_meta, f, indent=2)

                ckpt_model = ModelCheckpoint(
                    checkpoint_id=f"step-{step}",
                    step=step,
                    epoch=round(step / steps_per_epoch, 2),
                    loss=round(current_loss, 4),
                    eval_loss=round(current_loss * 1.05, 4),
                    checkpoint_path=str(ckpt_dir),
                    metrics={"loss": round(current_loss, 4)},
                    is_best=(step == total_steps),
                )
                checkpoints.append(ckpt_model)

        # Write dummy final weights
        dummy_weights_file = weights_dir / "model.safetensors"
        with open(dummy_weights_file, "w", encoding="utf-8") as f:
            f.write("mock_model_weights_tensor_binary_payload")

        dummy_config_file = weights_dir / "config.json"
        with open(dummy_config_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "architectures": [params.architecture],
                    "model_type": "qwen2",
                    "vocab_size": 151936,
                    "hidden_size": 3584,
                },
                f,
                indent=2,
            )

        metrics = TrainingMetrics(
            loss_history=loss_history,
            eval_loss_history=eval_loss_history,
            lr_history=lr_history,
            epoch_metrics=[
                {
                    "epoch": ep,
                    "loss": round(target_loss + (initial_loss - target_loss) * math.exp(-ep), 4),
                }
                for ep in range(1, total_epochs + 1)
            ],
        )

        return TrainingBackendResult(
            status="success",
            model_weights_dir=str(weights_dir),
            metrics=metrics,
            checkpoints=checkpoints,
            framework_metadata={
                "framework": "mock_backend",
                "simulated_steps": total_steps,
                "dataset_sample_count": samples_count,
            },
            execution_time_seconds=1.5,
        )
