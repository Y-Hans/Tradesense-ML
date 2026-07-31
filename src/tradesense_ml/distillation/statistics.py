"""Distillation Statistics generator consuming DistillationProcessingResult."""

from typing import Any

from tradesense_ml.domain.schemas.distillation import (
    DistillationProcessingResult,
    DistillationStatistics,
)


class StatisticsGenerator:
    """Generator computing reusable DistillationStatistics from DistillationProcessingResult."""

    @staticmethod
    def generate_statistics(
        processing_result: DistillationProcessingResult,
        **kwargs: Any,
    ) -> DistillationStatistics:
        """Compute aggregate statistics container."""
        sampled = processing_result.sampled_examples
        rejected = processing_result.rejected_examples
        pref_pairs = processing_result.preference_pairs
        curriculum = processing_result.curriculum_stages

        # 1. Selection & Rejection counts
        selection_counts = {
            "total_selected": len(processing_result.selected_examples),
            "total_sampled": len(sampled),
            "total_rejected": len(rejected),
        }
        rejection_counts = processing_result.filtering_stats.get("rejection_reasons", {})

        # 2. Sampling stats
        sampling_stats = {
            "strategy": processing_result.sampling_result.strategy_name,
            "sample_size": processing_result.sampling_result.sample_size,
            "sampling_rate": processing_result.sampling_result.sampling_rate,
        }

        # 3. Curriculum distribution
        curriculum_dist = {stage.name: stage.example_count for stage in curriculum}

        # 4. Preference counts
        preference_counts = {
            "total_pairs": len(pref_pairs),
        }

        # 5. Teacher, Difficulty, & Quality distributions
        teacher_dist: dict[str, int] = {}
        diff_dist: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0, "expert": 0}
        quality_dist: dict[str, int] = {"9.0-10.0": 0, "8.0-8.9": 0, "7.0-7.9": 0, "<7.0": 0}

        est_bytes = 0
        prompt_tokens = 0
        response_tokens = 0

        for ex in sampled:
            # Teacher
            t_id = ex.teacher_id or "teacher_llm_v1"
            teacher_dist[t_id] = teacher_dist.get(t_id, 0) + 1

            # Difficulty
            tier = ex.quality_tier.lower()
            diff_dist[tier] = diff_dist.get(tier, 0) + 1

            # Quality score bins
            sc = ex.quality_score
            if sc >= 9.0:
                quality_dist["9.0-10.0"] += 1
            elif sc >= 8.0:
                quality_dist["8.0-8.9"] += 1
            elif sc >= 7.0:
                quality_dist["7.0-7.9"] += 1
            else:
                quality_dist["<7.0"] += 1

            # Size and token estimations (approx 4 chars per token)
            text_size = len(ex.instruction) + len(ex.input) + len(ex.output) + len(ex.prompt)
            est_bytes += text_size
            prompt_tokens += (len(ex.instruction) + len(ex.input)) // 4
            response_tokens += len(ex.output) // 4

        for pair in pref_pairs:
            est_bytes += len(pair.chosen_response) + len(pair.rejected_response)

        token_estimates = {
            "prompt_tokens_est": prompt_tokens,
            "response_tokens_est": response_tokens,
            "total_tokens_est": prompt_tokens + response_tokens,
        }

        return DistillationStatistics(
            selection_counts=selection_counts,
            rejection_counts=rejection_counts,
            sampling_statistics=sampling_stats,
            curriculum_distribution=curriculum_dist,
            preference_counts=preference_counts,
            teacher_distribution=teacher_dist,
            difficulty_distribution=diff_dist,
            quality_distribution=quality_dist,
            dataset_size_bytes=est_bytes,
            total_examples=len(sampled),
            token_estimates=token_estimates,
        )
