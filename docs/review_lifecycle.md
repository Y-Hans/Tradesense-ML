# Dedicated Review Pipeline Architecture

The TradeSense ML review architecture ensures high data quality through a 4-stage review process before any synthetic or raw sample enters a training split.

---

## The 4 Review Stages

```text
[ Raw / Generated Sample ]
           │
           ▼
Stage 1: Automated Validation
 (Schema validity, required fields, reason code checks)
           │
           ▼
Stage 2: AI Teacher Review
 (Consensus evaluation against versioned rubric)
           │
           ▼
Stage 3: Human Review
 (Sampled audit queue for domain expert review)
           │
           ▼
Stage 4: Approval & Dataset Promotion
 (Promotion to approved training/validation dataset split)
```

---

## Audit Records

Every action across all stages records an immutable `ReviewAuditRecord`:
- `record_id`: Unique record ID
- `stage`: ReviewStage enum
- `reviewer_id`: Identifier of reviewer (system, AI teacher, human)
- `decision`: APPROVE, REJECT, NEEDS_REVISION, ESCALATE
- `score`: Quality score (0.0 - 10.0)
- `comments`: Justification / notes
- `timestamp`: UTC timestamp
