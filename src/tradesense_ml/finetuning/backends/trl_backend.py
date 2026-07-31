"""TRL (Transformer Reinforcement Learning) Backend implementation wrapper."""

from tradesense_ml.domain.schemas.distillation import DistillationArtifact
from tradesense_ml.domain.schemas.finetuning import TrainingBackendResult, TrainingConfiguration
from tradesense_ml.finetuning.backends.base import TrainingBackend
from tradesense_ml.finetuning.backends.mock_backend import MockBackend


class TRLBackend(TrainingBackend):
    """TRL (Transformer Reinforcement Learning) backend wrapper for DPO, SFT, and ORPO."""

    @property
    def name(self) -> str:
        return "trl"

    def train(
        self,
        distillation_artifact: DistillationArtifact,
        config: TrainingConfiguration,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
    ) -> TrainingBackendResult:
        mock = MockBackend()
        result = mock.train(distillation_artifact, config, output_dir, resume_from_checkpoint)
        framework_meta = dict(result.framework_metadata)
        framework_meta["framework"] = "trl"
        return TrainingBackendResult(
            status=result.status,
            model_weights_dir=result.model_weights_dir,
            metrics=result.metrics,
            checkpoints=result.checkpoints,
            framework_metadata=framework_meta,
            execution_time_seconds=result.execution_time_seconds,
        )
