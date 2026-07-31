"""Exporters for serializing DistillationArtifact into JSON, JSONL, Parquet, and Markdown formats."""

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tradesense_ml.domain.schemas.distillation import DistillationArtifact
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class BaseDistillationExporter(ABC):
    """Abstract base class for distillation artifact exporters."""

    def __init__(self, format_name: str) -> None:
        self.format_name = format_name

    @abstractmethod
    def export(
        self,
        artifact: DistillationArtifact,
        output_path: Path | str,
    ) -> Path:
        """Export DistillationArtifact to target output file path."""
        pass


class JSONDistillationExporter(BaseDistillationExporter):
    """Serializes complete canonical DistillationArtifact into formatted JSON."""

    def __init__(self) -> None:
        super().__init__(format_name="json")

    def export(
        self,
        artifact: DistillationArtifact,
        output_path: Path | str,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = artifact.model_dump_json(indent=2)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Exported DistillationArtifact '{artifact.artifact_id}' to JSON: {path}")
        return path


class JSONLDistillationExporter(BaseDistillationExporter):
    """Serializes SFT examples line-by-line into JSONL format."""

    def __init__(self) -> None:
        super().__init__(format_name="jsonl")

    def export(
        self,
        artifact: DistillationArtifact,
        output_path: Path | str,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write SFT examples if present, else preference pairs
        lines = []
        if artifact.dataset.sft_examples:
            lines = [ex.model_dump_json() for ex in artifact.dataset.sft_examples]
        elif artifact.dataset.preference_pairs:
            lines = [pair.model_dump_json() for pair in artifact.dataset.preference_pairs]

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Exported {len(lines)} records to JSONL: {path}")
        return path


class ParquetDistillationExporter(BaseDistillationExporter):
    """Serializes SFT examples or preference pairs to columnar Parquet format."""

    def __init__(self) -> None:
        super().__init__(format_name="parquet")

    def export(
        self,
        artifact: DistillationArtifact,
        output_path: Path | str,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        if artifact.dataset.sft_examples:
            records = [
                {
                    "example_id": ex.example_id,
                    "instruction": ex.instruction,
                    "input": ex.input,
                    "output": ex.output,
                    "quality_score": ex.quality_score,
                    "difficulty": ex.difficulty,
                    "quality_tier": ex.quality_tier,
                    "teacher_id": ex.teacher_id,
                }
                for ex in artifact.dataset.sft_examples
            ]
        elif artifact.dataset.preference_pairs:
            records = [
                {
                    "pair_id": pair.pair_id,
                    "example_id": pair.example_id,
                    "instruction": pair.instruction,
                    "input": pair.input,
                    "chosen_response": pair.chosen_response,
                    "rejected_response": pair.rejected_response,
                    "chosen_score": pair.chosen_score,
                    "rejected_score": pair.rejected_score,
                }
                for pair in artifact.dataset.preference_pairs
            ]

        try:
            import pandas as pd

            df = pd.DataFrame(records)
            df.to_parquet(path, index=False)
            logger.info(f"Exported {len(records)} records to Parquet: {path}")
            return path
        except ImportError:
            fallback_path = path.with_suffix(".json")
            logger.warning(
                f"Pandas/PyArrow unavailable for Parquet export; falling back to JSON: {fallback_path}"
            )
            fallback_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            return fallback_path


class MarkdownReportExporter(BaseDistillationExporter):
    """Generates rich Markdown report for DistillationArtifact."""

    def __init__(self) -> None:
        super().__init__(format_name="md")

    def export(
        self,
        artifact: DistillationArtifact,
        output_path: Path | str,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rpt = artifact.report
        meta = artifact.metadata
        summary = artifact.summary
        stats = artifact.statistics

        md_lines = [
            "# TradeSense ML Distillation Report",
            "",
            f"**Artifact ID**: `{meta.artifact_id}`  ",
            f"**Version**: `{meta.version}`  ",
            f"**Source Dataset**: `{meta.dataset_artifact_id}`  ",
            f"**Source Benchmark**: `{meta.benchmark_artifact_id}`  ",
            f"**Primary Teacher Model**: `{meta.teacher_model}`  ",
            f"**Execution Timestamp**: `{meta.created_at}`  ",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"- **Total Input Examples**: `{summary.total_input_examples}`",
            f"- **Selected Examples**: `{summary.total_selected_examples}`",
            f"- **Sampled SFT Dataset Size**: `{summary.total_sampled_examples}` examples (`{stats.dataset_size_bytes}` bytes)",
            f"- **Generated Preference Pairs**: `{summary.total_preference_pairs}`",
            f"- **Curriculum Stages**: `{summary.total_curriculum_stages}`",
            f"- **Mean Quality Score**: `{summary.overall_quality_mean:.2f} / 10.0`",
            f"- **Execution Latency**: `{summary.execution_time_seconds:.2f} seconds`",
            "",
            "### Curriculum Distribution",
            "",
            "| Stage | Count |",
            "| :--- | :---: |",
        ]

        for stage, count in stats.curriculum_distribution.items():
            md_lines.append(f"| `{stage}` | `{count}` |")

        md_lines.extend(
            [
                "",
                "### Quality Score Distribution",
                "",
                "| Quality Bin | Count |",
                "| :--- | :---: |",
            ]
        )
        for bin_name, count in stats.quality_distribution.items():
            md_lines.append(f"| `{bin_name}` | `{count}` |")

        if rpt.warnings:
            md_lines.extend(
                [
                    "",
                    "### Warnings",
                    "",
                ]
            )
            for warn in rpt.warnings:
                md_lines.append(f"- ⚠️ {warn}")

        md_lines.extend(
            [
                "",
                "---",
                "",
                "## Actionable Recommendations",
                "",
            ]
        )
        for rec in rpt.recommendations:
            md_lines.append(f"- 💡 {rec}")

        md_lines.extend(
            [
                "",
                "---",
                "",
                "## Lineage & Provenance",
                "",
                f"- **Configuration Hash**: `{artifact.lineage.configuration_hash}`",
                f"- **Distillation Strategy**: `{artifact.lineage.distillation_strategy}`",
                f"- **Selection Strategy**: `{artifact.lineage.selection_strategy}`",
                f"- **Sampling Strategy**: `{artifact.lineage.sampling_strategy}`",
                f"- **Curriculum Strategy**: `{artifact.lineage.curriculum_strategy}`",
                f"- **Random Seed**: `{artifact.lineage.random_seed}`",
                "",
            ]
        )

        content = "\n".join(md_lines)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Exported Distillation Markdown report to: {path}")
        return path


class DistillationExporterManager:
    """Manager orchestrating export of canonical DistillationArtifact across requested formats."""

    @classmethod
    def export_artifact(
        cls,
        artifact: DistillationArtifact,
        output_dir: Path | str,
        formats: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Export DistillationArtifact to output directory in requested formats.

        Args:
            artifact: Immutable canonical DistillationArtifact payload.
            output_dir: Base output directory path.
            formats: Desired formats (e.g. ['json', 'jsonl', 'parquet', 'md']).

        Returns:
            List of exported file descriptors with path, format, size_bytes, and SHA-256 checksum.
        """
        out_dir = Path(output_dir) / artifact.artifact_id
        out_dir.mkdir(parents=True, exist_ok=True)

        requested_formats = formats or ["json", "jsonl", "parquet", "md"]
        descriptors: list[dict[str, Any]] = []

        exporters = {
            "json": JSONDistillationExporter(),
            "jsonl": JSONLDistillationExporter(),
            "parquet": ParquetDistillationExporter(),
            "md": MarkdownReportExporter(),
        }

        for fmt in requested_formats:
            fmt_clean = fmt.lower().strip().replace(".", "")
            if fmt_clean in exporters:
                ext = "md" if fmt_clean == "md" else fmt_clean
                filename = f"{artifact.artifact_id}_distillation.{ext}"
                filepath = out_dir / filename

                exported_path = exporters[fmt_clean].export(artifact, filepath)
                file_bytes = exported_path.read_bytes()
                checksum = hashlib.sha256(file_bytes).hexdigest()

                descriptors.append(
                    {
                        "path": str(exported_path),
                        "format": fmt_clean,
                        "size_bytes": len(file_bytes),
                        "checksum": checksum,
                    }
                )

        return descriptors
