"""TrainingSession abstraction managing execution lifecycle state."""

import platform
from collections.abc import Callable
from datetime import datetime
from typing import Any

from tradesense_ml import __version__
from tradesense_ml.domain.schemas.distillation import DistillationArtifact
from tradesense_ml.domain.schemas.finetuning import (
    ModelCheckpoint,
    TrainingBackendResult,
    TrainingConfiguration,
    TrainingExecution,
)
from tradesense_ml.finetuning.backends.base import TrainingBackend
from tradesense_ml.finetuning.strategies import TrainingStrategy


class TrainingSession:
    """Session object owning the state and lifecycle of a single fine-tuning execution."""

    def __init__(
        self,
        run_id: str,
        distillation_artifact: DistillationArtifact,
        config: TrainingConfiguration,
        backend: TrainingBackend,
        strategy: TrainingStrategy,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.distillation_artifact = distillation_artifact
        self.config = config
        self.backend = backend
        self.strategy = strategy
        self.output_dir = output_dir

        # Session Lifecycle State
        self.active_checkpoint: ModelCheckpoint | None = None
        self.resume_state: str | None = resume_from_checkpoint
        self.current_epoch: float = 0.0
        self.current_step: int = 0
        self.optimizer_state: dict[str, Any] = {"optimizer": config.model_config_params.optimizer}
        self.scheduler_state: dict[str, Any] = {
            "scheduler": config.model_config_params.lr_scheduler
        }
        self.callbacks: list[Callable[[str, dict[str, Any]], None]] = []
        self.runtime_metadata: dict[str, Any] = {
            "session_id": run_id,
            "strategy": strategy.name,
            "backend": backend.name,
            "resumed": resume_from_checkpoint is not None,
        }

        # Hardware/Software Execution Context
        self.execution_context = TrainingExecution(
            python_version=platform.python_version(),
            pytorch_version="2.2.0+mock",
            cuda_version="12.1",
            gpu_models=["Mock NVIDIA H100 SXM5 80GB"],
            gpu_count=1,
            os_info=f"{platform.system()} {platform.release()}",
            git_commit="head-dev",
            repository_version=__version__,
            start_timestamp=datetime.utcnow(),
        )

    def register_callback(self, callback_fn: Callable[[str, dict[str, Any]], None]) -> None:
        """Register a lifecycle event callback handler."""
        self.callbacks.append(callback_fn)

    def trigger_callbacks(self, event: str, payload: dict[str, Any]) -> None:
        """Execute registered callbacks for a specific event trigger."""
        for cb in self.callbacks:
            try:
                cb(event, payload)
            except Exception:
                pass

    def update_progress(self, step: int, epoch: float, loss: float) -> None:
        """Update active session step, epoch, and loss metrics."""
        self.current_step = step
        self.current_epoch = epoch
        self.runtime_metadata["latest_loss"] = loss
        self.trigger_callbacks("on_step_end", {"step": step, "epoch": epoch, "loss": loss})

    def on_checkpoint_saved(self, checkpoint: ModelCheckpoint) -> None:
        """Track newly saved checkpoint in active session state."""
        self.active_checkpoint = checkpoint
        self.trigger_callbacks("on_checkpoint_save", {"checkpoint": checkpoint})

    def run(self) -> TrainingBackendResult:
        """Execute training strategy under active session lifecycle management."""
        self.trigger_callbacks("on_session_start", {"run_id": self.run_id})

        # Delegate workflow to training strategy
        result = self.strategy.execute(
            distillation_artifact=self.distillation_artifact,
            backend=self.backend,
            config=self.config,
            output_dir=self.output_dir,
            resume_from_checkpoint=self.resume_state,
        )

        # Finalize execution context metadata
        end_time = datetime.utcnow()
        duration = (end_time - self.execution_context.start_timestamp).total_seconds()

        self.execution_context = TrainingExecution(
            python_version=self.execution_context.python_version,
            pytorch_version=self.execution_context.pytorch_version,
            cuda_version=self.execution_context.cuda_version,
            gpu_models=self.execution_context.gpu_models,
            gpu_count=self.execution_context.gpu_count,
            os_info=self.execution_context.os_info,
            git_commit=self.execution_context.git_commit,
            repository_version=self.execution_context.repository_version,
            start_timestamp=self.execution_context.start_timestamp,
            end_timestamp=end_time,
            total_execution_seconds=duration,
        )

        if result.checkpoints:
            self.active_checkpoint = result.checkpoints[-1]

        self.trigger_callbacks("on_session_end", {"status": result.status})
        return result
