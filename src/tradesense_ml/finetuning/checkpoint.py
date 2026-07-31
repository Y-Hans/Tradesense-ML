"""Checkpoint Manager for checkpoint metadata, selection, validation, and resume support."""

import json
from pathlib import Path
from typing import Any

from tradesense_ml.domain.schemas.finetuning import CheckpointResult, ModelCheckpoint


class CheckpointManager:
    """Manages model checkpoint metadata, validation, resume paths, and best-checkpoint selection."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def register_checkpoint(
        self,
        step: int,
        epoch: float,
        loss: float,
        checkpoint_dir: str,
        eval_loss: float | None = None,
        metrics: dict[str, float] | None = None,
    ) -> ModelCheckpoint:
        """Register and save metadata for a training checkpoint."""
        ckpt_id = f"step-{step}"
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        ckpt_info: dict[str, Any] = {
            "checkpoint_id": ckpt_id,
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "eval_loss": eval_loss,
            "metrics": metrics or {},
        }
        with open(ckpt_path / "checkpoint_info.json", "w", encoding="utf-8") as f:
            json.dump(ckpt_info, f, indent=2)

        return ModelCheckpoint(
            checkpoint_id=ckpt_id,
            step=step,
            epoch=epoch,
            loss=loss,
            eval_loss=eval_loss,
            checkpoint_path=str(ckpt_path),
            metrics=metrics or {"loss": loss},
            is_best=False,
        )

    def select_best_checkpoint(self, checkpoints: list[ModelCheckpoint]) -> ModelCheckpoint | None:
        """Select best checkpoint based on eval_loss (or training loss if eval_loss unavailable)."""
        if not checkpoints:
            return None

        def key_fn(c: ModelCheckpoint) -> float:
            return c.eval_loss if c.eval_loss is not None else c.loss

        sorted_ckpts = sorted(checkpoints, key=key_fn)
        best = sorted_ckpts[0]

        # Return new instance with is_best=True
        return ModelCheckpoint(
            checkpoint_id=best.checkpoint_id,
            step=best.step,
            epoch=best.epoch,
            loss=best.loss,
            eval_loss=best.eval_loss,
            checkpoint_path=best.checkpoint_path,
            metrics=best.metrics,
            created_at=best.created_at,
            is_best=True,
        )

    def compile_checkpoint_result(
        self,
        checkpoints: list[ModelCheckpoint],
        resumed_from_path: str | None = None,
    ) -> CheckpointResult:
        """Compile a canonical CheckpointResult summary."""
        if not checkpoints:
            return CheckpointResult(
                total_checkpoints_saved=0,
                best_checkpoint=None,
                final_checkpoint=None,
                all_checkpoints=[],
                resumed_from_path=resumed_from_path,
            )

        best_ckpt = self.select_best_checkpoint(checkpoints)
        final_ckpt = checkpoints[-1]

        # Update all checkpoints list so best_ckpt is properly flagged
        updated_ckpts: list[ModelCheckpoint] = []
        for c in checkpoints:
            is_b = best_ckpt is not None and c.checkpoint_id == best_ckpt.checkpoint_id
            updated_ckpts.append(
                ModelCheckpoint(
                    checkpoint_id=c.checkpoint_id,
                    step=c.step,
                    epoch=c.epoch,
                    loss=c.loss,
                    eval_loss=c.eval_loss,
                    checkpoint_path=c.checkpoint_path,
                    metrics=c.metrics,
                    created_at=c.created_at,
                    is_best=is_b,
                )
            )

        return CheckpointResult(
            total_checkpoints_saved=len(checkpoints),
            best_checkpoint=best_ckpt,
            final_checkpoint=final_ckpt,
            all_checkpoints=updated_ckpts,
            resumed_from_path=resumed_from_path,
        )

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """Validate integrity of a checkpoint path."""
        p = Path(checkpoint_path)
        if not p.exists() or not p.is_dir():
            return False
        info_file = p / "checkpoint_info.json"
        return info_file.exists()
