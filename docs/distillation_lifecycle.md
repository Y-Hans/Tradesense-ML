# Distillation Lifecycle

Knowledge distillation allows distilling reasoning and coaching capability from larger frontier Teacher LLMs into compact, local Student LLMs.

---

## Distillation Pipeline Flow

```text
Teacher Models (Claude 3.5, GPT-4o, Gemini 1.5)
                 │
                 ▼
Synthetic & Curated Prompt Generation
                 │
                 ▼
Multi-Teacher Consensus & Output Generation
                 │
                 ▼
Filtering & Rubric Quality Scoring
                 │
                 ▼
Reviewed Training Dataset Packaging
                 │
                 ▼
Student Model Fine-Tuning (Qwen 2.5 / Llama 3)
                 │
                 ▼
Benchmark Evaluation & Registry
```
