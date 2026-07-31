"""Preference Builder for generating DPO/ORPO canonical PreferencePair datasets."""

from typing import Any

from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.domain.schemas.distillation import DistillationExample, PreferencePair
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class PreferenceBuilder:
    """Builder for generating DPO/ORPO canonical preference pairs from distillation and rejected examples."""

    def build_preference_pairs(
        self,
        chosen_examples: list[DistillationExample],
        rejected_examples: list[DistillationExample] | None = None,
        benchmark_artifact: BenchmarkArtifact | None = None,
        min_score_delta: float = 1.0,
        **kwargs: Any,
    ) -> list[PreferencePair]:
        """Build canonical PreferencePair list pairing chosen responses with rejected responses.

        Args:
            chosen_examples: List of high quality approved teacher outputs.
            rejected_examples: Optional list of lower-rated or flawed teacher outputs.
            benchmark_artifact: Optional benchmark artifact for scoring context.
            min_score_delta: Minimum required score gap between chosen and rejected.

        Returns:
            List of canonical PreferencePair objects.
        """
        pairs: list[PreferencePair] = []

        rejected_pool = list(rejected_examples) if rejected_examples else []

        bm_meta = {}
        if benchmark_artifact is not None:
            bm_meta = {
                "benchmark_id": benchmark_artifact.artifact_id,
                "overall_score": benchmark_artifact.scores.overall_score,
                "pass_rate": benchmark_artifact.summary.pass_rate,
            }

        # Case 1: Pair chosen with explicit rejected examples if available
        if rejected_pool:
            for i, chosen in enumerate(chosen_examples):
                rej = rejected_pool[i % len(rejected_pool)]
                delta = chosen.quality_score - rej.quality_score

                if delta < min_score_delta:
                    # Construct synthetic sub-optimal contrast response if delta is small
                    rejected_text = (
                        f"Trade Coaching Feedback:\n"
                        f"Your trade entry at {chosen.input.split('@')[-1] if '@' in chosen.input else 'market'} was fine. "
                        "Don't worry about stop losses or risk management right now, just hold until profit."
                    )
                    rejected_score = max(1.0, chosen.quality_score - 4.0)
                    rationale = "The rejected response ignores risk management rules and encourages unmanaged drawdown."
                else:
                    rejected_text = rej.output
                    rejected_score = rej.quality_score
                    rationale = f"Chosen response has quality score {chosen.quality_score:.1f} vs rejected score {rej.quality_score:.1f}."

                pair = PreferencePair(
                    pair_id=f"pref_{chosen.example_id}_{i:03d}",
                    example_id=chosen.example_id,
                    instruction=chosen.instruction,
                    input=chosen.input,
                    prompt=chosen.prompt,
                    chosen_response=chosen.output,
                    rejected_response=rejected_text,
                    preference_rationale=rationale,
                    chosen_score=chosen.quality_score,
                    rejected_score=rejected_score,
                    teacher_metadata={"teacher_id": chosen.teacher_id},
                    benchmark_metadata=bm_meta,
                    metadata={"min_score_delta": min_score_delta},
                )
                pairs.append(pair)

        # Case 2: If no explicit rejected pool provided, construct contrasting sub-optimal responses for DPO pairs
        else:
            for i, chosen in enumerate(chosen_examples):
                rejected_text = (
                    "Trade Coaching Feedback:\n"
                    "Good trade entry. You can ignore your planned stop loss level and wait for market recovery."
                )
                rejected_score = max(1.0, chosen.quality_score - 3.5)
                rationale = "Chosen response provides disciplined risk management rules; rejected response violates stop loss discipline."

                pair = PreferencePair(
                    pair_id=f"pref_{chosen.example_id}_{i:03d}",
                    example_id=chosen.example_id,
                    instruction=chosen.instruction,
                    input=chosen.input,
                    prompt=chosen.prompt,
                    chosen_response=chosen.output,
                    rejected_response=rejected_text,
                    preference_rationale=rationale,
                    chosen_score=chosen.quality_score,
                    rejected_score=rejected_score,
                    teacher_metadata={"teacher_id": chosen.teacher_id},
                    benchmark_metadata=bm_meta,
                    metadata={"synthetic_rejected": True},
                )
                pairs.append(pair)

        logger.info(f"PreferenceBuilder generated {len(pairs)} canonical PreferencePair objects.")
        return pairs
