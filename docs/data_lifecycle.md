# Data Lifecycle & Provenance Lineage

## Dataset Lifecycle Phases

1. **Scenario Generation**: Synthetic market context and trade history generation using scenario configs and cognitive bias injectors (`src/tradesense_ml/pipelines/generation/`).
2. **Teacher Inference**: Synthetic prompts sent to multi-teacher router (`src/tradesense_ml/teachers/router.py`) to generate ground-truth coaching responses.
3. **Automated Validation**: Pydantic schema validation, syntax checks, and reason code constraint verification (`src/tradesense_ml/pipelines/validation/`).
4. **Multi-Stage Review**: Automated -> AI Teacher -> Human Queue -> Approval (`src/tradesense_ml/pipelines/review/`).
5. **Dataset Versioning**: Final approved dataset assigned immutable semantic version and lineage metadata (`DatasetVersionMetadata`).

---

## Provenance Lineage Metadata

Every dataset produced in TradeSense ML stores complete provenance:

```json
{
  "dataset_id": "tradesense_coaching_v1",
  "dataset_version": "1.2.0",
  "parent_dataset_id": "synthetic_seed_v0",
  "teacher_model": "anthropic/claude-3.5-sonnet",
  "prompt_version": "v1",
  "rubric_version": "v1",
  "generator_version": "0.1.0",
  "review_version": "1.0.0",
  "generation_timestamp": "2026-07-30T12:00:00Z",
  "source_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "review_status": "APPROVED",
  "quality_score": 9.4
}
```
