---
title: "Clinical Named Entity Recognition"
summary: "Token-level BERT classifier that extracts disease and treatment entities from clinical text, evaluated with entity-level seqeval metrics."
tags: [NLP, BERT, NER, Clinical, Python]
order: 2
---

## Overview

Named Entity Recognition (NER) in clinical text is a foundational task for healthcare NLP. This project trains a token-level classifier to identify **disease** and **treatment** mentions in medical text — enabling downstream tasks like relation extraction, clinical coding, and patient phenotyping.

## Task Definition

Given a sequence of tokens, classify each token as:

| Label | Meaning | Example |
|-------|---------|---------|
| **O** | Outside (not an entity) | "the", "patient", "has" |
| **D** | Disease | "diabetes", "hypertension" |
| **T** | Treatment | "insulin", "aspirin" |

For evaluation, labels are converted to BIO format (`B-Disease`, `B-Treatment`) and scored at the **entity level** using seqeval — meaning "gestational diabetes" must be fully correct to count as a match.

## Pipeline

### 1. Data Loading

Training data is in CoNLL format — one token per line with blank lines separating sentences. The preprocessing script reconstructs sentences and aligns labels:

```
Patient    O
has        O
diabetes   D
treated    O
with       O
insulin    T
```

**Dataset size:** 48,501 training tokens across ~2,800 sentences; 19,674 test tokens.

### 2. Subword Alignment

BERT's WordPiece tokenizer splits words into subwords (e.g., "hypertension" becomes `["hyper", "##tension"]`). This creates a mismatch with word-level labels. The alignment strategy:

- **First subword** of each word receives the actual label
- **Continuation subwords** (`##` prefix) receive `-100`, which PyTorch's cross-entropy loss ignores automatically
- **Special tokens** (`[CLS]`, `[SEP]`, `[PAD]`) also receive `-100`

### 3. Model

- **Architecture:** `bert-base-uncased` with a token classification head (linear layer mapping each token's hidden state to 3 classes)
- **Hyperparameters:** learning rate 2e-5, batch size 16, 3 epochs, max sequence length 128
- **Model selection:** best checkpoint by F1 score on the test set

### 4. Inference

The `extract_entities()` function processes raw text and returns structured output:

```python
extract_entities("Patient has diabetes treated with insulin")

# Returns:
{
    "diseases": ["diabetes"],
    "treatments": ["insulin"]
}
```

The function handles subword reconstruction — reassembling `##` tokens back into complete words.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Token classification (not span extraction) | Simpler architecture; standard approach for NER with pre-trained transformers |
| `-100` masking for subwords | Prevents the model from being penalized on artificial token boundaries |
| Entity-level evaluation (seqeval) | More meaningful than token-level accuracy — partial entity matches don't count |
| bert-base-uncased | Clinical text is not case-sensitive for entity types; uncased reduces vocab complexity |

## Tools & Libraries

Python, PyTorch, Hugging Face Transformers, seqeval, pandas
