# Training Lifecycle & Adapter Management

TradeSense ML supports fine-tuning small student models (e.g. 7B/8B parameter models) using parameter-efficient fine-tuning (PEFT/LoRA/QLoRA) or full fine-tuning.

---

## Workflow

1. **Select Dataset**: Query Model Registry and Dataset Lineage for an `APPROVED` dataset split.
2. **Configure Run**: Modify `configs/training/default.yaml` or pass parameters to `tsml train`.
3. **Execute Fine-Tuning**: Launch training pipeline (`src/tradesense_ml/pipelines/training/base.py`).
4. **Checkpoint Management**: Save checkpoints with `BaseCheckpointManager`.
5. **Log to MLflow**: Log metrics, hyperparameter configs, and loss curves.
6. **Register Model**: Catalog trained checkpoint in Model Registry (`src/tradesense_ml/models/registry/base.py`).
