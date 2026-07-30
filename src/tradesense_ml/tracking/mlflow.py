"""MLflow experiment tracking abstraction wrapper."""

from typing import Any

from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class MLflowTracker:
    """Wrapper abstraction for MLflow experiment logging."""

    def __init__(
        self, experiment_name: str = "tradesense_default", tracking_uri: str = "outputs/mlruns"
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._mlflow: Any = None

    def initialize(self) -> None:
        """Initialize MLflow tracking URI and experiment."""
        try:
            import mlflow

            self._mlflow = mlflow
            self._mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow.set_experiment(self.experiment_name)
            logger.info(
                f"MLflow initialized: experiment='{self.experiment_name}', uri='{self.tracking_uri}'"
            )
        except ImportError:
            logger.warning("MLflow not installed. Tracking calls will operate in mock mode.")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameter key-values."""
        if self._mlflow and self._mlflow.active_run():
            self._mlflow.log_params(params)
        else:
            logger.info(f"[MLflow Mock] Log params: {params}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metric numerical values."""
        if self._mlflow and self._mlflow.active_run():
            self._mlflow.log_metrics(metrics, step=step)
        else:
            logger.info(f"[MLflow Mock] Log metrics (step={step}): {metrics}")

    def log_artifact(self, local_path: str) -> None:
        """Log artifact file path."""
        if self._mlflow and self._mlflow.active_run():
            self._mlflow.log_artifact(local_path)
        else:
            logger.info(f"[MLflow Mock] Log artifact: {local_path}")
