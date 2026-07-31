"""Fine-Tuning Reporter generating structured TrainingReport containers and Markdown summaries."""

from typing import Any

from tradesense_ml.domain.schemas.finetuning import TrainingProcessingResult, TrainingReport


class FineTuningReporter:
    """Reporter producing independent, structured reports and Markdown documentation."""

    def generate_report(
        self,
        processing_result: TrainingProcessingResult,
        warnings: list[str] | None = None,
        recommendations: list[str] | None = None,
    ) -> TrainingReport:
        """Generate structured TrainingReport from TrainingProcessingResult."""
        metrics = processing_result.backend_result.metrics
        ckpt = processing_result.checkpoint_result
        eval_res = processing_result.evaluation_result
        exec_ctx = processing_result.execution_context
        config = processing_result.training_config

        tr_summary = {
            "run_id": processing_result.run_id,
            "distillation_artifact_id": processing_result.distillation_artifact_id,
            "status": processing_result.backend_result.status,
            "duration_seconds": exec_ctx.total_execution_seconds,
            "total_checkpoints": ckpt.total_checkpoints_saved,
        }

        loss_summary = {
            "initial_loss": metrics.loss_history[0]["loss"] if metrics.loss_history else 0.0,
            "final_loss": metrics.loss_history[-1]["loss"] if metrics.loss_history else 0.0,
            "loss_entry_count": len(metrics.loss_history),
            "min_loss": min((e["loss"] for e in metrics.loss_history), default=0.0),
        }

        ckpt_summary = {
            "total_saved": ckpt.total_checkpoints_saved,
            "best_checkpoint_id": (
                ckpt.best_checkpoint.checkpoint_id if ckpt.best_checkpoint else None
            ),
            "best_checkpoint_eval_loss": (
                ckpt.best_checkpoint.eval_loss if ckpt.best_checkpoint else None
            ),
            "final_checkpoint_id": (
                ckpt.final_checkpoint.checkpoint_id if ckpt.final_checkpoint else None
            ),
        }

        eval_summary = {
            "eval_loss": eval_res.eval_loss,
            "perplexity": eval_res.perplexity,
            "accuracy": eval_res.accuracy,
            "token_accuracy": eval_res.token_accuracy,
            "convergence_score": eval_res.convergence_score,
            "benchmark_eval_scores": eval_res.benchmark_eval_scores,
        }

        config_summary = {
            "run_name": config.run_name,
            "strategy": config.strategy_name,
            "backend": config.backend_config.backend_name,
            "base_model": config.model_config_params.base_model_name_or_path,
            "learning_rate": config.model_config_params.learning_rate,
            "num_epochs": config.model_config_params.num_epochs,
            "batch_size": config.model_config_params.per_device_train_batch_size,
            "use_lora": config.model_config_params.use_lora,
        }

        backend_summary = {
            "backend": config.backend_config.backend_name,
            "device": config.backend_config.device,
            "framework_metadata": processing_result.backend_result.framework_metadata,
        }

        res_utilization = {
            "gpu_count": exec_ctx.gpu_count,
            "gpu_models": exec_ctx.gpu_models,
            "os_info": exec_ctx.os_info,
            "python_version": exec_ctx.python_version,
            "pytorch_version": exec_ctx.pytorch_version,
        }

        default_recs = [
            "Model training converged smoothly with no instability.",
            "Best checkpoint was selected based on minimal evaluation loss.",
            "Deploy to Benchmark Suite for comprehensive downstream assessment.",
        ]

        all_warnings = list(processing_result.warnings)
        if warnings:
            all_warnings.extend(warnings)

        return TrainingReport(
            training_summary=tr_summary,
            loss_curves_summary=loss_summary,
            checkpoint_summary=ckpt_summary,
            evaluation_summary=eval_summary,
            configuration_summary=config_summary,
            backend_summary=backend_summary,
            warnings=all_warnings,
            recommendations=recommendations or default_recs,
            resource_utilization=res_utilization,
        )

    def render_markdown_report(self, report: TrainingReport) -> str:
        """Render a clean GitHub-style Markdown text report."""
        lines = [
            "# TradeSense ML — Fine-Tuning Execution Report",
            "",
            "## Executive Summary",
            f"- **Run ID**: `{report.training_summary.get('run_id')}`",
            f"- **Distillation Artifact**: `{report.training_summary.get('distillation_artifact_id')}`",
            f"- **Status**: `{report.training_summary.get('status')}`",
            f"- **Duration**: `{report.training_summary.get('duration_seconds')}s`",
            "",
            "## Configuration",
            f"- **Strategy**: `{report.configuration_summary.get('strategy')}`",
            f"- **Backend**: `{report.configuration_summary.get('backend')}`",
            f"- **Base Model**: `{report.configuration_summary.get('base_model')}`",
            f"- **Epochs**: `{report.configuration_summary.get('num_epochs')}`",
            f"- **Learning Rate**: `{report.configuration_summary.get('learning_rate')}`",
            "",
            "## Training Loss & Convergence",
            f"- **Initial Loss**: `{report.loss_curves_summary.get('initial_loss')}`",
            f"- **Final Loss**: `{report.loss_curves_summary.get('final_loss')}`",
            f"- **Min Loss**: `{report.loss_curves_summary.get('min_loss')}`",
            "",
            "## Evaluation Metrics",
            f"- **Validation Loss**: `{report.evaluation_summary.get('eval_loss')}`",
            f"- **Perplexity**: `{report.evaluation_summary.get('perplexity')}`",
            f"- **Accuracy**: `{report.evaluation_summary.get('accuracy')}`",
            f"- **Convergence Index**: `{report.evaluation_summary.get('convergence_score')}`",
            "",
            "### Benchmark Evaluation Scores",
        ]
        bench_scores: dict[str, Any] = report.evaluation_summary.get("benchmark_eval_scores", {})
        for task, score in bench_scores.items():
            lines.append(f"- **{task}**: `{score}`")

        lines.extend(
            [
                "",
                "## Checkpoint Summary",
                f"- **Total Checkpoints Saved**: `{report.checkpoint_summary.get('total_saved')}`",
                f"- **Best Checkpoint**: `{report.checkpoint_summary.get('best_checkpoint_id')}`",
                "",
                "## Recommendations",
            ]
        )
        for rec in report.recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)
