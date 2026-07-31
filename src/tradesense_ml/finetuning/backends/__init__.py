"""Training backends package exports and auto-registration."""

from tradesense_ml.finetuning.backends.axolotl_backend import AxolotlBackend
from tradesense_ml.finetuning.backends.base import BackendRegistry, TrainingBackend
from tradesense_ml.finetuning.backends.huggingface_backend import HuggingFaceBackend
from tradesense_ml.finetuning.backends.mock_backend import MockBackend
from tradesense_ml.finetuning.backends.trl_backend import TRLBackend
from tradesense_ml.finetuning.backends.unsloth_backend import UnslothBackend

# Register built-in backends automatically
BackendRegistry.register("mock", MockBackend)
BackendRegistry.register("unsloth", UnslothBackend)
BackendRegistry.register("axolotl", AxolotlBackend)
BackendRegistry.register("huggingface", HuggingFaceBackend)
BackendRegistry.register("trl", TRLBackend)

__all__ = [
    "TrainingBackend",
    "BackendRegistry",
    "MockBackend",
    "UnslothBackend",
    "AxolotlBackend",
    "HuggingFaceBackend",
    "TRLBackend",
]
