# TradeSense ML Repository Philosophy: The AI Factory

## Core Belief

> **The primary enterprise asset is NOT fine-tuned model checkpoint weights — it is the proprietary data ecosystem surrounding them.**

Open-source base models (e.g. Gemma, Qwen, Llama, Mistral) rapidly improve. Today's state-of-the-art model is superseded tomorrow. If a research system binds its code tightly to a specific model architecture or training script, every model upgrade forces a rewrite.

TradeSense ML is designed as a long-term **AI Factory** where:
- **Datasets** carry complete cryptographic and provenance lineage.
- **Prompts & Rubrics** live as versioned, declarative assets outside of code.
- **Teacher Models** are pluggable providers accessible via unified interfaces.
- **Evaluation** is conducted via reproducible benchmark suites against strict domain rubrics.
- **Student Models** are interchangeable commodities that consume distilled datasets.

---

## Architectural Principles

1. **Decouple Data & Prompts from Code**: Prompts, rubrics, and templates are stored under `assets/` and versioned independently.
2. **Provider Agnosticism**: No code assumes a single provider. OpenRouter, OpenAI, Anthropic, Gemini, Ollama, and local HuggingFace/vLLM share the `BaseTeacherProvider` interface.
3. **Strict Lineage Provenance**: Every generated dataset record tracks its parent dataset, prompt version, teacher model ID, rubric version, generator version, and cryptographic hash.
4. **Multi-Stage Review Pipelines**: Data must earn its place in training splits by passing through automated validation, AI teacher consensus, and human audit queues.
5. **No Business Logic in CLI or Notebooks**: Core logic lives in `src/tradesense_ml/`. Notebooks in `research/` and commands in `src/tradesense_ml/cli/` are thin execution wrappers.
