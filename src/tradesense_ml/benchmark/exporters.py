"""Exporters for serializing BenchmarkArtifact objects into JSON, JSONL, Parquet, and Markdown formats."""

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tradesense_ml.domain.schemas.benchmark import BenchmarkArtifact
from tradesense_ml.logging.logger import get_logger

logger = get_logger()


class BaseBenchmarkExporter(ABC):
    """Abstract base class for benchmark artifact exporters."""

    def __init__(self, format_name: str) -> None:
        self.format_name = format_name

    @abstractmethod
    def export(
        self,
        artifact: BenchmarkArtifact,
        output_path: Path | str,
    ) -> Path:
        """Export BenchmarkArtifact to target output file path."""
        pass


class JSONBenchmarkExporter(BaseBenchmarkExporter):
    """Serializes complete canonical BenchmarkArtifact into formatted JSON."""

    def __init__(self) -> None:
        super().__init__(format_name="json")

    def export(
        self,
        artifact: BenchmarkArtifact,
        output_path: Path | str,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = artifact.model_dump_json(indent=2)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Exported BenchmarkArtifact '{artifact.artifact_id}' to JSON: {path}")
        return path


class JSONLBenchmarkExporter(BaseBenchmarkExporter):
    """Serializes benchmark results line-by-line into JSONL format."""

    def __init__(self) -> None:
        super().__init__(format_name="jsonl")

    def export(
        self,
        artifact: BenchmarkArtifact,
        output_path: Path | str,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [res.model_dump_json() for res in artifact.results]
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Exported {len(artifact.results)} benchmark results to JSONL: {path}")
        return path


class ParquetBenchmarkExporter(BaseBenchmarkExporter):
    """Serializes benchmark metric table to columnar Parquet format."""

    def __init__(self) -> None:
        super().__init__(format_name="parquet")

    def export(
        self,
        artifact: BenchmarkArtifact,
        output_path: Path | str,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metric_records = [
            {
                "artifact_id": artifact.artifact_id,
                "case_id": res.case_id,
                "case_name": res.case_name,
                "concern": res.concern,
                "score": res.score,
                "weight": res.weight,
                "passed": res.passed,
                "metric_id": m.metric_id,
                "metric_name": m.name,
                "metric_type": m.metric_type,
                "metric_value": m.value,
                "unit": m.unit,
            }
            for res in artifact.results
            for m in res.metrics
        ]

        try:
            import pandas as pd

            df = pd.DataFrame(metric_records)
            df.to_parquet(path, index=False)
            logger.info(
                f"Exported benchmark metrics table ({len(metric_records)} rows) to Parquet: {path}"
            )
            return path
        except ImportError:
            fallback_path = path.with_suffix(".json")
            logger.warning(
                f"Pandas/PyArrow unavailable for Parquet export; falling back to JSON: {fallback_path}"
            )
            fallback_path.write_text(json.dumps(metric_records, indent=2), encoding="utf-8")
            return fallback_path


class MarkdownReportExporter(BaseBenchmarkExporter):
    """Generates rich, human-readable GitHub-style Markdown report."""

    def __init__(self) -> None:
        super().__init__(format_name="md")

    def export(
        self,
        artifact: BenchmarkArtifact,
        output_path: Path | str,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rpt = artifact.report
        meta = artifact.metadata
        summary = artifact.summary
        scores = artifact.scores

        md_lines = [
            "# TradeSense ML Benchmark Evaluation Report",
            "",
            f"**Benchmark ID**: `{meta.benchmark_id}`  ",
            f"**Suite Name**: `{meta.suite_name}` (`{artifact.profile.name}`)  ",
            f"**Target Model**: `{meta.target_model}`  ",
            f"**Dataset Release**: `{meta.dataset_id}:{meta.dataset_version}`  ",
            f"**Execution Timestamp**: `{meta.created_at}`  ",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"- **Overall Benchmark Score**: `{summary.overall_score:.2f} / 10.0`",
            f"- **Ranking Tier**: **{scores.ranking_info.get('tier', 'N/A')}** (Grade `{scores.ranking_info.get('grade', 'N/A')}`)",
            f"- **Pass Rate**: `{summary.pass_rate * 100:.1f}%` ({summary.passed_cases}/{summary.total_cases} cases passed)",
            f"- **Execution Time**: `{summary.execution_time_seconds:.2f} seconds`",
            "",
            "### Category Scores",
            "",
            "| Category | Score / 10.0 | Status |",
            "| :--- | :---: | :---: |",
        ]

        for cat, sc in scores.category_scores.items():
            status_emoji = "🟢 PASS" if sc >= 6.0 else "🔴 FAIL"
            md_lines.append(f"| `{cat}` | `{sc:.2f}` | {status_emoji} |")

        md_lines.extend(
            [
                "",
                "---",
                "",
                "## Benchmark Case Breakdown",
                "",
                "| Case ID | Name | Concern | Weight | Score | Verdict |",
                "| :--- | :--- | :--- | :---: | :---: | :---: |",
            ]
        )

        for res in artifact.results:
            verdict = "🟢 PASS" if res.passed else "🔴 FAIL"
            md_lines.append(
                f"| `{res.case_id}` | {res.case_name} | {res.concern} | `{res.weight}` | `{res.score:.2f}` | {verdict} |"
            )

        if rpt.failures:
            md_lines.extend(
                [
                    "",
                    "### Failures",
                    "",
                ]
            )
            for fail in rpt.failures:
                md_lines.append(
                    f"- **{fail['case_name']}** (`{fail['case_id']}`): {', '.join(fail['failure_reasons'])}"
                )

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
                "## Lineage & Environment Provenance",
                "",
                f"- **Configuration Hash**: `{artifact.lineage.configuration_hash}`",
                f"- **Random Seed**: `{artifact.lineage.random_seed}`",
                f"- **Benchmark Engine Version**: `{artifact.lineage.benchmark_version}`",
                f"- **Prompt Template Version**: `{artifact.lineage.prompt_version}`",
                "",
            ]
        )

        content = "\n".join(md_lines)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Exported Benchmark Markdown report to: {path}")
        return path


class BenchmarkExporterManager:
    """Manager orchestrating export of canonical BenchmarkArtifact across requested formats."""

    @classmethod
    def export_artifact(
        cls,
        artifact: BenchmarkArtifact,
        output_dir: Path | str,
        formats: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Export BenchmarkArtifact to output directory in requested formats.

        Args:
            artifact: Immutable canonical BenchmarkArtifact payload.
            output_dir: Base output directory path.
            formats: Desired formats (e.g. ['json', 'jsonl', 'parquet', 'md']).

        Returns:
            List of exported file descriptors with path, format, and SHA-256 checksum.
        """
        out_dir = Path(output_dir) / artifact.artifact_id
        out_dir.mkdir(parents=True, exist_ok=True)

        requested_formats = formats or ["json", "jsonl", "parquet", "md"]
        descriptors: list[dict[str, Any]] = []

        exporters = {
            "json": JSONBenchmarkExporter(),
            "jsonl": JSONLBenchmarkExporter(),
            "parquet": ParquetBenchmarkExporter(),
            "md": MarkdownReportExporter(),
        }

        for fmt in requested_formats:
            fmt_clean = fmt.lower().strip().replace(".", "")
            if fmt_clean in exporters:
                ext = "md" if fmt_clean == "md" else fmt_clean
                filename = f"{artifact.artifact_id}_benchmark.{ext}"
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
