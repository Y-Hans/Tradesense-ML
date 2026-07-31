"""Distillation Runner coordinating strategy execution, policies, and retries."""

import time
from typing import Any

from tradesense_ml.distillation.strategies import DistillationStrategyRegistry
from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.dataset import DatasetArtifact
from tradesense_ml.domain.schemas.distillation import DistillationProcessingResult
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class DistillationRunner:
    """Dedicated runner delegating execution to DistillationStrategy while managing policies and retries."""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    def run_strategy(
        self,
        dataset_artifact: DatasetArtifact,
        benchmark_artifact: BenchmarkArtifact | None = None,
        strategy_name: str = "SFTStrategy",
        config_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DistillationProcessingResult:
        """Coordinate execution using specified DistillationStrategy.

        Args:
            dataset_artifact: Input DatasetArtifact release.
            benchmark_artifact: Optional BenchmarkArtifact release.
            strategy_name: Name of registered DistillationStrategy.
            config_dict: Execution options and parameters.

        Returns:
            DistillationProcessingResult intermediate container.
        """
        strategy = DistillationStrategyRegistry.get(strategy_name)
        cfg = config_dict or {}

        attempt = 0
        last_exception: Exception | None = None

        while attempt <= self.max_retries:
            try:
                attempt += 1
                logger.info(
                    f"DistillationRunner executing strategy '{strategy_name}' (attempt {attempt}/{self.max_retries + 1})"
                )
                start_time = time.perf_counter()

                result = strategy.execute(
                    dataset_artifact=dataset_artifact,
                    benchmark_artifact=benchmark_artifact,
                    config_dict=cfg,
                    **kwargs,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                logger.info(
                    f"DistillationRunner completed strategy '{strategy_name}' in {latency_ms:.2f}ms. "
                    f"Selected: {len(result.selected_examples)}, Sampled: {len(result.sampled_examples)}, "
                    f"Preference Pairs: {len(result.preference_pairs)}"
                )
                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt} for strategy '{strategy_name}' failed with error: {e}"
                )
                if attempt > self.max_retries:
                    break
                time.sleep(0.1)

        raise RuntimeError(
            f"DistillationRunner failed strategy '{strategy_name}' after {attempt} attempts. Last error: {last_exception}"
        ) from last_exception
