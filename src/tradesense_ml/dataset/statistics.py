"""Dataset statistics computation module."""

import math

from tradesense_ml.domain.schemas.dataset import DatasetExample, DatasetStatistics
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class DatasetStatisticsGenerator:
    """Statistics generator calculating quality metrics, text lengths, distributions, and byte sizes."""

    @classmethod
    def generate(
        cls,
        dataset_id: str,
        examples: list[DatasetExample],
        total_evaluated: int | None = None,
        rejected_count: int = 0,
        split_sizes: dict[str, int] | None = None,
        version_info: dict[str, str] | None = None,
    ) -> DatasetStatistics:
        """Compute aggregate statistics for a set of DatasetExample objects."""
        tot_eval = (
            total_evaluated if total_evaluated is not None else (len(examples) + rejected_count)
        )
        approved_count = len(examples)

        if not examples:
            return DatasetStatistics(
                dataset_id=dataset_id,
                total_examples=tot_eval,
                approved_examples=0,
                rejected_examples=rejected_count,
                split_sizes=split_sizes or {},
                version_info=version_info or {},
            )

        scores: list[float] = []
        confidences: list[float] = []
        resp_lengths: list[int] = []
        prompt_lengths: list[int] = []
        total_bytes = 0

        teacher_dist: dict[str, int] = {}
        reviewer_dist: dict[str, int] = {}

        for ex in examples:
            # Estimate record size in bytes
            dump_str = ex.model_dump_json()
            total_bytes += len(dump_str.encode("utf-8"))

            prompt_lengths.append(len(ex.prompt))
            resp_lengths.append(len(ex.output))

            # Quality metrics from review_info
            if ex.review_info:
                q_score = ex.review_info.get("quality_score")
                if isinstance(q_score, (int, float)):
                    scores.append(float(q_score))

                conf = ex.review_info.get("confidence")
                if isinstance(conf, (int, float)):
                    confidences.append(float(conf))

                rev_name = str(ex.review_info.get("reviewer_name", "unknown"))
                reviewer_dist[rev_name] = reviewer_dist.get(rev_name, 0) + 1

            # Lineage tracking
            if ex.lineage:
                t_model = str(
                    ex.lineage.get("teacher_model", ex.lineage.get("teacher_provider", "unknown"))
                )
                teacher_dist[t_model] = teacher_dist.get(t_model, 0) + 1

        score_mean = sum(scores) / float(len(scores)) if scores else 0.0
        score_min = min(scores) if scores else 0.0
        score_max = max(scores) if scores else 0.0

        if len(scores) > 1:
            variance = sum((s - score_mean) ** 2 for s in scores) / float(len(scores) - 1)
            score_std = math.sqrt(variance)
        else:
            score_std = 0.0

        avg_conf = sum(confidences) / float(len(confidences)) if confidences else 1.0
        avg_resp_len = sum(resp_lengths) / float(len(resp_lengths)) if resp_lengths else 0.0
        avg_prompt_len = sum(prompt_lengths) / float(len(prompt_lengths)) if prompt_lengths else 0.0

        stats = DatasetStatistics(
            dataset_id=dataset_id,
            total_examples=tot_eval,
            approved_examples=approved_count,
            rejected_examples=rejected_count,
            quality_score_mean=round(score_mean, 2),
            quality_score_min=round(score_min, 2),
            quality_score_max=round(score_max, 2),
            quality_score_std=round(score_std, 2),
            average_confidence=round(avg_conf, 2),
            average_response_length=round(avg_resp_len, 1),
            average_prompt_length=round(avg_prompt_len, 1),
            dataset_size_bytes=total_bytes,
            split_sizes=split_sizes or {},
            teacher_distribution=teacher_dist,
            reviewer_distribution=reviewer_dist,
            version_info=version_info or {},
        )

        logger.info(
            f"Generated statistics for dataset '{dataset_id}': {approved_count} approved examples, mean score={stats.quality_score_mean}"
        )
        return stats
