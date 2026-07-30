"""Model export and serving deployment interfaces."""

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field


class ExportConfig(BaseModel):
    """Model export configuration (e.g. GGUF, ONNX, merged HF checkpoint)."""

    model_id: str = Field(..., description="Target model ID in registry")
    export_format: str = Field(default="gguf", description="gguf, onnx, or safetensors")
    quantization: str | None = Field(default="q4_k_m", description="Quantization level")
    output_path: str = Field(default="./outputs/exports", description="Export target path")


class BaseDeploymentPipeline(ABC):
    """Pipeline for exporting and packaging models for Flutter / edge serving."""

    @abstractmethod
    def export(self, config: ExportConfig) -> Path:
        """Export model checkpoint to target format."""
        pass
