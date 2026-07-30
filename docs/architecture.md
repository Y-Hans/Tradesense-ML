# TradeSense ML Architecture Overview

TradeSense ML is structured into modular layers aligned with the AI lifecycle.

```text
                                  +-----------------------+
                                  |   Typer CLI (tsml)    |
                                  +-----------+-----------+
                                              |
                                              v
+-----------------------+         +-----------+-----------+
|  Hydra Config System  |<------->|    AI Pipelines       |
|      (configs/)       |         | (src/tradesense_ml/   |
+-----------------------+         |    pipelines/)        |
                                  +-----------+-----------+
                                              |
     +-------------------+--------------------+--------------------+
     |                   |                    |                    |
     v                   v                    v                    v
+----+----+        +-----+----+         +-----+----+         +-----+----+
| Assets  |        | Teachers |         | Domain   |         | Student  |
| Manager |        | Router   |         | Schemas  |         | Models & |
|(assets/)|        |(teachers/|         |(schemas/)|         | Registry |
+---------+        +----------+         +----------+         +----------+
```

---

## Folder Breakdown

- `assets/`: Declarative versioned prompts, evaluation rubrics, market templates, seed examples, and benchmark definitions.
- `configs/`: Hydra configuration hierarchy (`config.yaml`, `dataset/`, `teacher/`, `training/`, `evaluation/`, `model/`, `logging/`, `storage/`, `experiment/`).
- `docs/`: System documentation and lifecycle guides.
- `research/`: Notebooks and experimental code sandbox.
- `src/tradesense_ml/`: Core python package.
  - `cli/`: Typer CLI command entrypoints.
  - `domain/schemas/`: Immutable Pydantic v2 schemas for all domain entities.
  - `teachers/`: Teacher provider implementations and consensus router.
  - `pipelines/`: AI lifecycle orchestrators (`ingestion`, `generation`, `validation`, `review`, `distillation`, `training`, `evaluation`, `deployment`).
  - `models/`: Student model interfaces, adapter management, and model registry.
  - `tracking/`: MLflow integration.
  - `storage/`: Abstract storage backends.
  - `logging/`: Centralized Loguru logger.
