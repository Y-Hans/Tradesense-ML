# TradeSense ML Extensibility Guide

This guide provides step-by-step instructions for extending the platform.

---

## How to Add a New Teacher Model Provider

1. Create a new provider file in `src/tradesense_ml/teachers/providers/my_provider.py`.
2. Inherit from `BaseTeacherProvider`.
3. Implement `_do_generate(self, request: TeacherRequest)`.
4. Export the class in `src/tradesense_ml/teachers/providers/__init__.py`.
5. Register the provider with `TeacherRouter.register_provider()`.

---

## How to Add a New Prompt Asset

1. Add text file under `assets/prompts/<version>/<prompt_name>.txt`.
2. Load it in python using `AssetManager().get_prompt("<prompt_name>", version="<version>")`.

---

## How to Add a New Evaluation Rubric

1. Add JSON file under `assets/rubrics/<version>/<rubric_name>.json`.
2. Follow the `Rubric` Pydantic model structure.
3. Load it using `AssetManager().get_rubric("<rubric_name>", version="<version>")`.

---

## How to Add a New Benchmark Suite

1. Add JSON file under `assets/benchmarks/<version>/<benchmark_name>.json`.
2. Load it using `AssetManager().get_benchmark("<benchmark_name>", version="<version>")`.
