"""Centralized BenchmarkScoringEngine separating raw case measurements from evaluation scoring."""

from typing import Any

from tradesense_ml.benchmark.cases import CaseRegistry
from tradesense_ml.benchmark.metrics import MetricRegistry
from tradesense_ml.domain.schemas.benchmark import (
    BenchmarkExecutionResult,
    BenchmarkMetric,
    BenchmarkProfile,
    BenchmarkResult,
    BenchmarkScore,
    BenchmarkSummary,
)


class BenchmarkScoringEngine:
    """Centralized scoring engine evaluating raw observations into standardized benchmark scores and rankings."""

    def __init__(self, default_pass_threshold: float = 6.0) -> None:
        self.default_pass_threshold = default_pass_threshold

    def evaluate(
        self,
        execution_results: list[BenchmarkExecutionResult],
        profile: BenchmarkProfile,
        pass_threshold: float | None = None,
    ) -> tuple[list[BenchmarkResult], list[BenchmarkMetric], BenchmarkScore, BenchmarkSummary]:
        """Process raw execution results into scored BenchmarkResults, metrics, scores, and summary.

        Args:
            execution_results: Raw un-scored execution results from runner.
            profile: Declarative benchmark profile containing scoring weights.
            pass_threshold: Optional score threshold for pass/fail verdict (default 6.0 / 10.0).

        Returns:
            Tuple of (scored BenchmarkResults, all BenchmarkMetrics, BenchmarkScore, BenchmarkSummary).
        """
        threshold = pass_threshold if pass_threshold is not None else self.default_pass_threshold
        case_weights_map = profile.case_weights or {}

        scored_results: list[BenchmarkResult] = []
        all_metrics: list[BenchmarkMetric] = []

        total_cases = len(execution_results)
        passed_cases = 0
        failed_cases = 0

        total_weighted_case_score = 0.0
        total_case_weight = 0.0

        category_score_sums: dict[str, float] = {}
        category_weight_sums: dict[str, float] = {}
        score_breakdown: dict[str, dict[str, Any]] = {}

        for exec_res in execution_results:
            case_id = exec_res.case_id

            # Lookup case definition or use default metadata
            if case_id in CaseRegistry.list_cases():
                case_def = CaseRegistry.get(case_id).get_definition()
                case_name = case_def.name
                concern = case_def.concern
                default_weight = case_def.weight
            else:
                case_name = case_id.replace("_", " ").title()
                concern = "general"
                default_weight = 1.0

            weight = case_weights_map.get(case_id, default_weight)

            # Derive case score (0.0 to 10.0) from raw metrics
            raw_metrics = exec_res.raw_metrics
            if "quality_score" in raw_metrics:
                raw_score = float(raw_metrics["quality_score"])
            elif "accuracy" in raw_metrics:
                raw_score = float(raw_metrics["accuracy"]) * 10.0
            elif "pass_rate" in raw_metrics:
                raw_score = float(raw_metrics["pass_rate"]) * 10.0
            elif "consistency_score" in raw_metrics:
                raw_score = float(raw_metrics["consistency_score"])
            else:
                raw_score = 7.0 if exec_res.status == "completed" else 0.0

            score = max(0.0, min(10.0, round(raw_score, 2)))
            passed = (score >= threshold) and (exec_res.status == "completed")

            if passed:
                passed_cases += 1
            else:
                failed_cases += 1

            # Convert raw metrics into canonical BenchmarkMetric objects
            case_metrics: list[BenchmarkMetric] = []
            for metric_key, metric_val in raw_metrics.items():
                scoped_id = f"{case_id}.{metric_key}"
                if metric_key in MetricRegistry.list_metrics():
                    base_m = MetricRegistry.get(metric_key).compute([metric_val])
                    metric_obj = BenchmarkMetric(
                        metric_id=scoped_id,
                        name=f"{case_name} - {base_m.name}",
                        metric_type=base_m.metric_type,
                        value=base_m.value,
                        unit=base_m.unit,
                        min_value=base_m.min_value,
                        max_value=base_m.max_value,
                        metadata={"case_id": case_id, **base_m.metadata},
                    )
                else:
                    metric_obj = BenchmarkMetric(
                        metric_id=scoped_id,
                        name=f"{case_name} - {metric_key.replace('_', ' ').title()}",
                        metric_type=metric_key,
                        value=float(metric_val),
                        metadata={"case_id": case_id},
                    )
                case_metrics.append(metric_obj)
                all_metrics.append(metric_obj)

            failure_reasons = []
            if not passed:
                if exec_res.error_message:
                    failure_reasons.append(exec_res.error_message)
                else:
                    failure_reasons.append(
                        f"Score ({score:.2f}) fell below minimum pass threshold ({threshold:.2f})."
                    )

            warnings = []
            if score < 7.5 and passed:
                warnings.append(f"Score ({score:.2f}) is close to pass threshold.")

            res = BenchmarkResult(
                case_id=case_id,
                case_name=case_name,
                concern=concern,
                passed=passed,
                score=score,
                weight=weight,
                metrics=case_metrics,
                details={
                    "latency_ms": exec_res.latency_ms,
                    "items_evaluated": exec_res.total_items_evaluated,
                    "failed_items": exec_res.failed_items_count,
                    "status": exec_res.status,
                },
                failure_reasons=failure_reasons,
                warnings=warnings,
            )
            scored_results.append(res)

            # Weight accumulation
            total_weighted_case_score += score * weight
            total_case_weight += weight

            # Grouping into scoring categories
            cat_name = self._map_concern_to_category(concern)
            category_score_sums[cat_name] = category_score_sums.get(cat_name, 0.0) + (
                score * weight
            )
            category_weight_sums[cat_name] = category_weight_sums.get(cat_name, 0.0) + weight

            score_breakdown[case_id] = {
                "name": case_name,
                "score": score,
                "weight": weight,
                "passed": passed,
                "category": cat_name,
            }

        # Calculate category scores
        category_scores: dict[str, float] = {}
        for cat, weight_sum in category_weight_sums.items():
            if weight_sum > 0:
                category_scores[cat] = round(category_score_sums[cat] / weight_sum, 2)
            else:
                category_scores[cat] = 0.0

        # Calculate overall weighted benchmark score
        overall_score = (
            round(total_weighted_case_score / total_case_weight, 2)
            if total_case_weight > 0
            else 0.0
        )

        # Ranking tier assessment
        if overall_score >= 9.0:
            ranking_tier = "Exceptional (Tier 1)"
            rating_grade = "A+"
        elif overall_score >= 7.5:
            ranking_tier = "Strong (Tier 2)"
            rating_grade = "A"
        elif overall_score >= 6.0:
            ranking_tier = "Adequate (Tier 3)"
            rating_grade = "B"
        else:
            ranking_tier = "Needs Improvement (Tier 4)"
            rating_grade = "C"

        ranking_info = {
            "tier": ranking_tier,
            "grade": rating_grade,
            "overall_score": overall_score,
            "pass_rate": round(passed_cases / max(1, total_cases), 4),
        }

        benchmark_score = BenchmarkScore(
            overall_score=overall_score,
            weighted_score=overall_score,
            category_scores=category_scores,
            score_breakdown=score_breakdown,
            ranking_info=ranking_info,
        )

        benchmark_summary = BenchmarkSummary(
            benchmark_id=f"bm_eval_{profile.profile_id}",
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            pass_rate=round(passed_cases / max(1, total_cases), 4),
            overall_score=overall_score,
            category_scores=category_scores,
            execution_time_seconds=sum(r.latency_ms for r in execution_results) / 1000.0,
            ranking_tier=ranking_tier,
        )

        return scored_results, all_metrics, benchmark_score, benchmark_summary

    @staticmethod
    def _map_concern_to_category(concern: str) -> str:
        """Map case concern string to scoring category."""
        c = concern.lower()
        if "risk" in c or "discipline" in c:
            return "risk_discipline"
        elif "coaching" in c or "pedagogical" in c:
            return "coaching"
        elif "consistency" in c or "completeness" in c or "quality" in c:
            return "consistency_quality"
        elif "safety" in c or "guardrail" in c:
            return "safety_compliance"
        elif "prompt" in c or "adherence" in c or "instruction" in c:
            return "formatting_compliance"
        return "general"
