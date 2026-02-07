---
title: "CAD Detection in Clinical Notes"
summary: "Fine-tuned Bio_ClinicalBERT with LoRA adapters to detect coronary artery disease mentions in clinical notes, using 5-fold cross-validation on MIMIC discharge summaries."
tags: [NLP, BERT, LoRA, Clinical, Python]
order: 1
---

## Overview

Coronary artery disease (CAD) is one of the leading causes of death globally. Identifying CAD mentions in unstructured clinical notes is critical for downstream clinical decision support, cohort identification, and epidemiological research.

This project builds a binary classifier that reads a patient's **Past Medical History** section from a clinical discharge summary and predicts whether the patient has CAD.

## Pipeline

### 1. Preprocessing

Clinical notes from the MIMIC database are processed through two stages:

- **PMH Extraction** — A regex-based parser isolates the Past Medical History section from full discharge summaries, using section headers (Social History, Family History, Allergies, etc.) as boundaries.

- **Label Assignment** — CAD-positive labels are assigned by matching clinical indicators:

```python
patterns = [
    r'\bcad\b',
    r'coronary artery disease',
    r'\bcabg\b',
    r'myocardial infarction',
    r's/p mi\b',
]
```

Negation detection prevents false positives from phrases like "no CAD", "denies CAD", or "ruled out CAD". Family history sections are also excluded to avoid attributing a family member's diagnosis to the patient.

After preprocessing: **8,877 labeled samples** from the original 206,943 records.

### 2. Model Architecture

- **Base model:** `emilyalsentzer/Bio_ClinicalBERT` — a BERT model pre-trained on clinical text from MIMIC-III
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) via the PEFT library
  - Rank: 8, Alpha: 16, Dropout: 0.1
  - Target modules: query and value attention matrices
  - Trains only ~0.3% of total parameters while keeping the base model frozen

### 3. Training

- **Validation:** Stratified 5-fold cross-validation
- **Hyperparameters:** learning rate 2e-4, batch size 8, 5 epochs per fold
- **Metric for model selection:** F1 score
- **Optimizer:** AdamW (Hugging Face Trainer default)

### 4. Inference

The trained model exposes a `classify_note()` function that takes raw clinical text and returns:

```python
{
    "prediction": "CAD" or "No CAD",
    "probability_cad": 0.94,
    "confidence": 0.94
}
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Bio_ClinicalBERT over base BERT | Pre-trained on clinical text; understands medical abbreviations and note structure |
| LoRA over full fine-tuning | Drastically reduces trainable parameters and training time while maintaining performance |
| PMH section extraction | Reduces noise — full notes contain labs, imaging, and other irrelevant sections |
| Negation handling | Clinical notes frequently mention conditions in negative context ("no history of CAD") |

## Tools & Libraries

Python, PyTorch, Hugging Face Transformers, PEFT, scikit-learn, pandas
