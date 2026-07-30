# TradeSense ML Coding Standards

1. **Type Hints Everywhere**: All function signatures must include strict Python type hints.
2. **Pydantic v2 Models**: Immutable domain models use `model_config = ConfigDict(frozen=True)`.
3. **Dependency Inversion**: High-level modules depend on abstractions (interfaces), not concrete details.
4. **Loguru Logging**: Use `get_logger()` from `tradesense_ml.logging.logger`.
5. **Hydra Configuration**: Store default parameters in YAML configs under `configs/`.
6. **Linting & Formatting**: Clean compliance with `ruff check .`, `black --check .`, and `mypy src`.
