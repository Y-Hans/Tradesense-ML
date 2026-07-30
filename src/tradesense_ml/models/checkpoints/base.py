"""Checkpoint manager interface."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseCheckpointManager(ABC):
    """Abstract interface for managing model checkpoints."""

    def __init__(self, checkpoint_dir: str = "./outputs/checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save_checkpoint(self, model_id: str, step: int, weights_dict: dict) -> Path:
        """Save a model checkpoint to disk."""
        pass

    @abstractmethod
    def load_latest_checkpoint(self, model_id: str) -> Path:
        """Find and return latest checkpoint for model_id."""
        pass
