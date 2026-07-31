"""BenchmarkRunner abstraction handling suite execution, ordering, retries, and policy enforcement."""

from typing import Any

from tradesense_ml.benchmark.suites import SuiteRegistry
from tradesense_ml.domain.schemas.benchmark import BenchmarkExecutionResult, BenchmarkProfile
from tradesense_ml.domain.schemas.dataset import DatasetArtifact
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class BenchmarkRunner:
    """Execution engine that executes benchmark suites according to profile and policy rules."""

    def __init__(self, retries: int = 0, fail_fast: bool = False) -> None:
        self.retries = retries
        self.fail_fast = fail_fast

    def run_profile(
        self,
        dataset_artifact: DatasetArtifact,
        profile: BenchmarkProfile,
        kwargs: dict[str, Any] | None = None,
    ) -> list[BenchmarkExecutionResult]:
        """Execute all suites and cases specified in a BenchmarkProfile.

        Args:
            dataset_artifact: Input canonical DatasetArtifact.
            profile: Declarative BenchmarkProfile specifying target suites and cases.
            kwargs: Extra runtime parameters.

        Returns:
            List of un-scored BenchmarkExecutionResult objects.
        """
        logger.info(
            f"BenchmarkRunner starting profile execution for '{profile.name}' ({profile.profile_id})"
        )

        execution_results: list[BenchmarkExecutionResult] = []
        policy = profile.execution_policy or {}
        max_retries = int(policy.get("retries", self.retries))

        for suite_name in profile.suite_names:
            if suite_name not in SuiteRegistry.list_suites():
                logger.warning(
                    f"Suite '{suite_name}' requested in profile '{profile.profile_id}' is not registered. Skipping."
                )
                continue

            suite = SuiteRegistry.get(suite_name)
            enabled_cases = profile.enabled_case_ids or None

            logger.info(
                f"Running suite '{suite.name}' ({suite.suite_id}) with enabled cases: {enabled_cases}"
            )

            attempt = 0
            while attempt <= max_retries:
                try:
                    suite_results = suite.run_cases(
                        dataset_artifact=dataset_artifact,
                        enabled_case_ids=enabled_cases,
                        kwargs=kwargs,
                    )
                    execution_results.extend(suite_results)
                    break
                except Exception as e:
                    attempt += 1
                    logger.error(
                        f"Error executing suite '{suite.suite_id}' (attempt {attempt}/{max_retries + 1}): {e}"
                    )
                    if attempt > max_retries:
                        if self.fail_fast or policy.get("fail_fast", False):
                            raise
                        # Return failed result placeholder
                        execution_results.append(
                            BenchmarkExecutionResult(
                                case_id=f"failed_{suite.suite_id}",
                                suite_id=suite.suite_id,
                                status="failed",
                                error_message=str(e),
                            )
                        )

        logger.info(
            f"BenchmarkRunner completed profile execution. Total raw case execution results collected: {len(execution_results)}"
        )
        return execution_results
