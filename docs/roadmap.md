# TradeSense ML AI Factory Roadmap

## Phase 1: Architecture & Scaffolding (Current Milestone) ✅
- Scalable `src/tradesense_ml` package layout
- Versioned `assets/` directory (prompts, rubrics, templates, benchmarks)
- Strongly typed Pydantic v2 schemas with complete dataset lineage
- Provider-agnostic Teacher model abstractions & multi-teacher router
- 4-stage Review Pipeline architecture (Automated -> AI -> Human -> Approval)
- Typer CLI (`tsml`) & Hydra config system
- Testing & verification setup (Ruff, Black, Mypy, Pytest)

## Phase 2: Synthetic Data & Teacher Integration 🔜
- Implement synthetic price series and trade generation algorithms
- Implement OpenRouter & OpenAI REST API teacher connectors
- Execute first 1,000-sample synthetic dataset generation run

## Phase 3: Review Pipeline Execution & Distillation 🔮
- Connect human annotation UI/queue to review pipeline
- Build HuggingFace TRL & PEFT QLoRA fine-tuning scripts
- Distill 7B/8B student model on approved coaching dataset

## Phase 4: Benchmarking, Export & Flutter Integration 🚀
- Execute reproducible benchmark runs
- Export quantized model to GGUF format
- Integrate ONNX/GGUF serving with TradeSense Flutter backend
