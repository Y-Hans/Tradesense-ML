# TradeSense ML 🚀

> **TradeSense AI Factory & Research Platform**
> Decoupled architecture for trading dataset generation, multi-teacher LLM evaluation, prompt engineering, review pipelines, student model distillation, and model registry.

---

## Vision & Philosophy

TradeSense ML is designed around a fundamental principle: **The primary enterprise asset is NOT any fine-tuned model checkpoint — it is the proprietary ecosystem surrounding it**:

- 📊 **Curated Trading Datasets** with full provenance lineage
- 📜 **Versioned System Prompts** & generation templates
- 🎯 **Domain-Specific Evaluation Rubrics** & reason code taxonomies
- 🔄 **Multi-Stage Review Pipelines** (Automated → AI Consensus → Human Review → Approval)
- 🏋️ **First-Class Reproducible Benchmark Suites**
- 🧠 **Provider-Agnostic Teacher Models** (OpenRouter, OpenAI, Anthropic, Gemini, Local LLMs)
- 📦 **Distillation & Student Fine-Tuning Infrastructure**

---

## Quickstart

### Installation

```bash
# Clone and setup with uv (or standard venv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode
uv pip install -e ".[dev,ml]"
```

### Command Line Interface (`tsml`)

```bash
# Display CLI commands
tsml --help

# Generate synthetic trade scenarios
tsml dataset generate --help

# Run multi-teacher generation
tsml teacher generate --help

# Start review pipeline queue
tsml review queue

# Fine-tune student model
tsml train --help

# Run evaluation benchmark
tsml benchmark run --help

# Model Registry
tsml registry list
```

---

## Repository Layout

```text
tradesense-ml/
├── assets/          # Declarative versioned prompts, rubrics, templates, benchmarks
├── configs/         # Hydra configuration files
├── docs/            # Platform architecture and lifecycle documentation
├── research/        # Notebooks and exploratory prototypes
├── scripts/         # Utility scripts
├── src/tradesense_ml/ # Core library source code
└── tests/           # Unit and integration test suite
```

For complete documentation, see [`docs/architecture.md`](docs/architecture.md) and [`docs/repository_philosophy.md`](docs/repository_philosophy.md).
