"""FineTuningRunner coordinating execution flow, sessions, retries, and resume support."""

import time
from typing import Any

from tradesense_ml.domain.schemas.distillation import DistillationArtifact
from tradesense_ml.domain.schemas.finetuning import TrainingBackendResult, TrainingConfiguration
from tradesense_ml.finetuning.backends.base import BackendRegistry
from tradesense_ml.finetuning.session import TrainingSession
from tradesense_ml.finetuning.strategies import TrainingStrategyRegistry


class FineTuningRunner:
    """Runner coordinating TrainingSession, strategy resolution, backend resolution, and retries."""

    def __init__(self, max_retries: int = 2, retry_delay_seconds: float = 1.0) -> None:
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def run(
        self,
        run_id: str,
        distillation_artifact: DistillationArtifact,
        config: TrainingConfiguration,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
        callback: Any | None = None,
    ) -> tuple[TrainingBackendResult, TrainingSession]:
        """Execute fine-tuning run via TrainingSession and resolve backend + strategy.

        Args:
            run_id: Unique run ID.
            distillation_artifact: Input DistillationArtifact.
            config: Execution configuration.
            output_dir: Output directory.
            resume_from_checkpoint: Optional checkpoint path.
            callback: Optional lifecycle callback.

        Returns:
            Tuple of (TrainingBackendResult, TrainingSession).
        """
        # Resolve backend and strategy from registries
        backend_name = config.backend_config.backend_name
        strategy_name = config.strategy_name

        backend = BackendRegistry.get(backend_name)
        strategy = TrainingStrategyRegistry.get(strategy_name)

        session = TrainingSession(
            run_id=run_id,
            distillation_artifact=distillation_artifact,
            config=config,
            backend=backend,
            strategy=strategy,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        if callback:
            session.register_callback(callback)

        attempt = 0
        last_exception: Exception | None = None

        while attempt <= self.max_retries:
            try:
                result = session.run()
                if result.status == "success":
                    return result, session
                attempt += 1
            except Exception as e:
                last_exception = e
                attempt += 1
                if attempt <= self.max_retries:
                    time.sleep(self.retry_delay_seconds)

        if last_exception:
            raise RuntimeError(
                f"Fine-tuning run '{run_id}' failed after {self.max_retries + 1} attempts: {last_exception}"
            ) from last_exception

        raise RuntimeError(
            f"Fine-tuning run '{run_id}' failed after {self.max_retries + 1} attempts."
        )
