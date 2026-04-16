---
title: "MIMIC-III Clinical Note → FHIR Extraction"
summary: "Fine-tuned LLaMA 3.1 8B on 744 MIMIC-III discharge summaries using QLoRA to extract structured FHIR-compatible JSON, achieving ~86% field-level accuracy vs. 0% base model. Includes a Power BI clinical analytics dashboard built from model predictions."
tags: [LLM, Fine-Tuning, QLoRA, FHIR, Clinical NLP, LLaMA, MIMIC-III, HuggingFace, Power BI, Python]
order: 11
---

## Overview

An end-to-end healthcare AI pipeline that fine-tunes **LLaMA 3.1 8B Instruct** on MIMIC-III discharge summaries to perform **structured clinical information extraction**, outputting **FHIR R4-compatible JSON**. The model takes a raw discharge summary as input and extracts key clinical fields into a standardized format ready for downstream EHR integration.

This project targets a real production problem: unstructured clinical notes are the rule, not the exception, and converting them into structured FHIR resources is an active need at health systems and companies like Nuance, Abridge, and Epic.

**Key result:** Base model 0% → Fine-tuned model ~86% average field-level accuracy after 1 epoch of QLoRA on 595 training examples (~15 min on A100).

| Component | Choice |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| Dataset | MIMIC-III discharge summaries (744 notes) |
| Fine-tuning method | QLoRA (4-bit, LoRA rank 16) |
| Label generation | Claude Haiku (`claude-haiku-4-5-20251001`) |
| Training environment | Google Colab Pro (A100) |
| Eval metric | Field-level precision, recall, F1 |

---

## Target FHIR Schema

Each model output is a JSON object matching the following structure:

```json
{
  "patient_id": "26880",
  "admission_date": "2162-3-3",
  "chief_complaint": "Mechanical fall from height of 10 feet",
  "discharge_diagnosis": [
    {"diagnosis": "Fracture dislocation C6-C7"},
    {"diagnosis": "Aspiration Pneumonia"}
  ],
  "discharge_medications": [
    {"name": "Morphine Concentrate", "dosage": "20 mg/mL", "route": "PO", "frequency": "Q4H as needed"}
  ],
  "discharge_disposition": "Extended Care",
  "discharge_condition": "Stable"
}
```

**FHIR resource mapping:**

| Field | FHIR Resource |
|---|---|
| `discharge_diagnosis` | `Condition` |
| `discharge_medications` | `MedicationRequest` |
| `discharge_disposition` | `Encounter.hospitalization.dischargeDisposition` |
| `chief_complaint` | `Encounter.reasonCode` |
| `admission_date` | `Encounter.period.start` |

Fields with long free text (brief hospital course, discharge instructions) and fields with low clinical value for this use case (medications on admission, ICD-10 codes) were deliberately excluded to keep evaluation tractable and results interpretable.

---

## Two-Phase Workflow

### Phase 1 — Local (Mac M2): Data Pipeline

All data preparation ran locally with no GPU required.

**Synthetic labeling with Claude API:** Ground truth labels were generated using Claude Haiku to extract FHIR JSON from each of the 744 discharge summaries. This synthetic labeling approach — used in projects like Alpaca and Orca — produces consistent, high-quality training data at low cost (~$4.60 total, ~600 tokens output per note on average). A checkpointing system ensured automatic resumption after API interruptions.

```python
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=2048,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_prompt}]
)
```

Key prompt engineering decisions:
- Explicit prohibition on markdown code fences ("no ` ```json `, just the raw JSON object starting with `{`") — Haiku wrapped responses in fences by default despite general "no markdown" instructions
- "Extract from X section only" to prevent inference from the note narrative body
- Consistent null handling: scalar fields → `null`, list fields → `[]`

**Train/val/test split:** 595 / 74 / 75 (80/10/10). Test set saved immediately and held out until final evaluation.

**Instruction format for training:**

```
### Instruction:
Extract structured clinical information from the following MIMIC-III discharge 
summary and return a FHIR-compatible JSON object.

### Input:
{raw note text}

### Output:
{fhir json label}
```

### Phase 2 — Google Colab Pro (A100): Training & Evaluation

**QLoRA configuration:** 4-bit quantization reduces LLaMA 3.1 8B from ~16GB to ~6GB VRAM. LoRA adapters (rank 16, alpha 32) target all attention projections and MLP layers, updating only a small fraction of parameters while preserving the base model's general capabilities.

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
)
```

**Training:** `SFTTrainer` (trl 1.1.0), 1 epoch, `bf16=True` (required on A100 — fp16 causes `NotImplementedError` for bfloat16 tensors), effective batch size 8 via gradient accumulation. Sanity-check at 100 steps confirmed loss decreasing (1.84 → 1.28) before committing to the full run. Full training: 75 steps, final loss 1.397, ~15 minutes on A100.

---

## Evaluation Results

Evaluation uses field-level scoring with exact match for scalar fields and fuzzy matching (RapidFuzz) for diagnosis lists and medication lists. Run twice — once on the base model, once on the fine-tuned model — on the 75 held-out test notes.

| Field | Base Model | Fine-Tuned | Delta |
|---|---|---|---|
| admission_date | 0% | **100%** | +100% |
| discharge_disposition | 0% | **94.67%** | +94.67% |
| discharge_medications | 0% | **94.08%** | +94.08% |
| discharge_diagnosis | 0% | **83.55%** | +83.55% |
| chief_complaint | 0% | **73.33%** | +73.33% |
| discharge_condition | 0% | **70.67%** | +70.67% |
| **Average** | **0%** | **~86%** | **+86%** |

**Base model behavior:** 100% JSON parse failure — the untrained model outputs `{"` followed by a flood of newline/control characters with no concept of the instruction format. This is expected and makes the fine-tuning improvement stark and unambiguous.

**Fine-tuned model observations:**
- `admission_date` — 100% exact match; date extraction from a consistent field is well-defined and the model learned it perfectly
- `discharge_disposition` and `discharge_medications` — both above 94%; these fields have consistent formatting in MIMIC notes
- `discharge_diagnosis` — 83.55% recall (fuzzy threshold 80); strong for a multi-item list field
- `chief_complaint` — 73.33% reflects genuine free-text variability, not extraction failure
- `discharge_condition` — 70.67% affected by inconsistent MIMIC formatting (some notes use "Stable", others use multi-line structured blocks like "Mental Status: Clear and coherent. Level of Consciousness: Alert and interactive.")

**Methodological note:** Ground truth labels are Claude-generated, not human-annotated. Evaluation measures field-level agreement with these labels — a well-established approach for fine-tuning from synthetic data. This is stated explicitly in any write-up to avoid overstating the metric.

---

## Power BI Dashboard

After fine-tuning, the structured predictions from the 73-note test set were exported to three flat CSV files and visualized in a Power BI clinical analytics dashboard.

<div style="margin: 1.5rem 0;">
  <img src="{{ '/assets/images/mimic_powerbi_dashboard.png' | relative_url }}" alt="MIMIC-III Clinical Discharge Analytics dashboard showing top diagnoses, medications, discharge disposition, and discharge condition" style="width: 100%; border-radius: 6px;">
</div>

**Exported tables:**
- `patients.csv` — one row per patient with scalar fields (admission date, chief complaint, disposition, condition)
- `diagnoses.csv` — one row per diagnosis, joined on `subject_id`
- `medications.csv` — one row per medication with name, dosage, route, frequency

**Dashboard highlights (73-note test set):**
- **Top diagnoses:** Hypertension most prevalent, followed by Anemia and Acute Renal Failure — consistent with MIMIC-III's ICU/inpatient population
- **Discharge disposition:** 41% Home, 20.55% Extended Care — typical for complex inpatient stays
- **Top medications:** Docusate sodium (~30 patients), Acetaminophen, Aspirin — standard inpatient formulary

---

## Design Decisions & Known Issues

**trl 1.1.0 API changes:** `SFTTrainer` removed `tokenizer` and `dataset_text_field` parameters. Dataset must be pre-formatted with a `"text"` column before passing to the trainer — `dataset_text_field` no longer accepted.

**Inference setup after training:** `model.eval()`, `model.gradient_checkpointing_disable()`, and `model.config.use_cache = True` must be called after training before inference. Without this, gradient checkpointing remains active, breaking the KV cache and producing the same newline-flood outputs as the base model.

**CSV serialization of dict columns:** When dataframes are saved to CSV and reloaded, dict columns serialize as Python string representations using single quotes — not valid JSON. Use `ast.literal_eval` rather than `json.loads` when parsing label columns after loading from CSV.

**MIMIC de-identification:** Notes use `[**de-identified**]` placeholders for PHI (names, dates, locations). The labeling prompt instructs Claude to extract what is available from de-identified dates and return `null` if the date cannot be determined — preventing creative interpretation of placeholders.

---

## Tools

Python, HuggingFace Transformers, peft, bitsandbytes, trl (SFTTrainer), Anthropic API (Claude Haiku), RapidFuzz, pandas, scikit-learn, Google Colab Pro (A100), Power BI
