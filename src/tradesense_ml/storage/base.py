"""Storage backend interface definitions for local, DuckDB, SQLite, S3, GCS, etc."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseStorageBackend(ABC, Generic[T]):
    """Abstract storage backend for datasets, artifacts, and metadata records."""

    @abstractmethod
    def save(self, key: str, record: T) -> None:
        """Save a record with a given key."""
        pass

    @abstractmethod
    def load(self, key: str, model_cls: type[T]) -> T:
        """Load a record by key and deserialize into Pydantic model."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in storage backend."""
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List stored keys matching optional prefix."""
        pass


class LocalDiskStorageBackend(BaseStorageBackend[T]):
    """Local filesystem implementation of storage backend."""

    def __init__(self, root_path: str = "./outputs/storage") -> None:
        from pathlib import Path

        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, record: T) -> None:
        target = self.root_path / f"{key}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def load(self, key: str, model_cls: type[T]) -> T:
        target = self.root_path / f"{key}.json"
        if not target.exists():
            raise FileNotFoundError(f"Key '{key}' not found at {target}")
        return model_cls.model_validate_json(target.read_text(encoding="utf-8"))

    def exists(self, key: str) -> bool:
        return (self.root_path / f"{key}.json").exists()

    def list_keys(self, prefix: str = "") -> list[str]:
        return [p.stem for p in self.root_path.glob(f"{prefix}*.json")]
