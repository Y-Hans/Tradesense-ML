"""Assets manager for resolving, loading, and versioning prompts, rubrics, templates, and benchmarks."""

import json
from pathlib import Path
from typing import Any, cast

from tradesense_ml.domain.schemas.rubrics import Rubric


class AssetManager:
    """Manager for loading versioned assets from the top-level assets directory."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            # Default to top-level assets/ directory relative to workspace root
            self.root_dir = Path(__file__).resolve().parents[3] / "assets"
        else:
            self.root_dir = Path(root_dir)

    def get_prompt(self, name: str, version: str = "v1") -> str:
        """Load text content of a system or evaluation prompt."""
        prompt_path = self.root_dir / "prompts" / version / f"{name}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    def get_rubric(self, name: str, version: str = "v1") -> Rubric:
        """Load a Rubric Pydantic model from rubrics directory."""
        rubric_path = self.root_dir / "rubrics" / version / f"{name}.json"
        if not rubric_path.exists():
            raise FileNotFoundError(f"Rubric file not found: {rubric_path}")
        data = json.loads(rubric_path.read_text(encoding="utf-8"))
        return Rubric.model_validate(data)

    def get_benchmark(self, name: str, version: str = "v1") -> dict[str, Any]:
        """Load benchmark suite definition."""
        benchmark_path = self.root_dir / "benchmarks" / version / f"{name}.json"
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
        res = json.loads(benchmark_path.read_text(encoding="utf-8"))
        return cast(dict[str, Any], res)

    def get_template(self, name: str) -> dict[str, Any]:
        """Load a JSON scenario template."""
        template_path = self.root_dir / "templates" / f"{name}.json"
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        res = json.loads(template_path.read_text(encoding="utf-8"))
        return cast(dict[str, Any], res)
