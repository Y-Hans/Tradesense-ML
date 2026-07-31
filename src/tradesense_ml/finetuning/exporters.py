"""Model Exporters writing ModelArtifact and ModelPackage outputs to disk."""

import json
from pathlib import Path
from typing import Any

from tradesense_ml.domain.schemas.finetuning import ModelArtifact


class ModelExporter:
    """Exporter orchestrating serializations into directory, JSON, Markdown, Manifest, and HF formats."""

    def export(
        self,
        artifact: ModelArtifact,
        output_dir: str,
        export_formats: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Export ModelArtifact contents in specified target formats.

        Args:
            artifact: Canonical ModelArtifact.
            output_dir: Target output root directory.
            export_formats: List of formats ("directory", "json", "markdown", "manifest", "huggingface").

        Returns:
            List of export file descriptors with paths and checksums.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        formats = export_formats or ["directory", "json", "markdown", "manifest"]

        descriptors: list[dict[str, Any]] = []

        if "directory" in formats:
            dir_desc = self._export_directory(artifact, out_path / "model_export")
            descriptors.append(dir_desc)

        if "json" in formats:
            json_desc = self._export_json(artifact, out_path / f"{artifact.artifact_id}.json")
            descriptors.append(json_desc)

        if "markdown" in formats:
            md_desc = self._export_markdown(
                artifact, out_path / f"{artifact.artifact_id}_report.md"
            )
            descriptors.append(md_desc)

        if "manifest" in formats:
            man_desc = self._export_manifest(
                artifact, out_path / f"{artifact.artifact_id}_manifest.json"
            )
            descriptors.append(man_desc)

        if "huggingface" in formats:
            hf_desc = self._export_huggingface_format(artifact, out_path / "huggingface_repo")
            descriptors.append(hf_desc)

        return descriptors

    def _export_directory(self, artifact: ModelArtifact, target_dir: Path) -> dict[str, Any]:
        target_dir.mkdir(parents=True, exist_ok=True)
        meta_file = target_dir / "model_artifact.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(artifact.model_dump(mode="json"), f, indent=2, default=str)
        return {
            "format": "directory",
            "path": str(target_dir),
            "checksum": artifact.package.package_checksum,
        }

    def _export_json(self, artifact: ModelArtifact, target_file: Path) -> dict[str, Any]:
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(artifact.model_dump(mode="json"), f, indent=2, default=str)
        return {
            "format": "json",
            "path": str(target_file),
            "checksum": artifact.package.package_checksum,
        }

    def _export_markdown(self, artifact: ModelArtifact, target_file: Path) -> dict[str, Any]:
        from tradesense_ml.finetuning.reporting import FineTuningReporter

        reporter = FineTuningReporter()
        md_content = reporter.render_markdown_report(artifact.report)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        return {
            "format": "markdown",
            "path": str(target_file),
            "checksum": artifact.package.package_checksum,
        }

    def _export_manifest(self, artifact: ModelArtifact, target_file: Path) -> dict[str, Any]:
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(artifact.manifest.model_dump(mode="json"), f, indent=2, default=str)
        return {
            "format": "manifest",
            "path": str(target_file),
            "checksum": artifact.manifest.package_checksum,
        }

    def _export_huggingface_format(
        self, artifact: ModelArtifact, target_dir: Path
    ) -> dict[str, Any]:
        target_dir.mkdir(parents=True, exist_ok=True)
        card_file = target_dir / "README.md"
        with open(card_file, "w", encoding="utf-8") as f:
            f.write(
                f"---\ntags:\n- tradesense-ml\n- coaching-llm\n---\n# {artifact.metadata.model_name}\n"
            )
        return {
            "format": "huggingface",
            "path": str(target_dir),
            "checksum": artifact.package.package_checksum,
        }
