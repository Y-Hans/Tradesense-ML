"""Teacher provider implementations."""

from tradesense_ml.teachers.providers.anthropic import AnthropicTeacherProvider
from tradesense_ml.teachers.providers.gemini import GeminiTeacherProvider
from tradesense_ml.teachers.providers.local import LocalTeacherProvider
from tradesense_ml.teachers.providers.ollama import OllamaTeacherProvider
from tradesense_ml.teachers.providers.openai import OpenAITeacherProvider
from tradesense_ml.teachers.providers.openrouter import OpenRouterTeacherProvider
from tradesense_ml.teachers.providers.vllm import VLLMTeacherProvider

__all__ = [
    "OpenRouterTeacherProvider",
    "OpenAITeacherProvider",
    "AnthropicTeacherProvider",
    "GeminiTeacherProvider",
    "LocalTeacherProvider",
    "OllamaTeacherProvider",
    "VLLMTeacherProvider",
]
