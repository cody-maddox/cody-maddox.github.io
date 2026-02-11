---
title: "Drug-Drug Interaction Extraction"
summary: "Fine-tuned BERT for relation extraction to classify drug-drug interactions from the DDI Corpus 2013, achieving 0.79 macro F1 across 5 interaction types."
tags: [NLP, BERT, Relation Extraction, Biomedical, Python]
order: 3
---

## Overview

Drug-drug interactions (DDIs) are a major cause of adverse drug events, and manually curating DDI databases is slow and incomplete. This project fine-tunes BERT for relation extraction to classify drug-drug interactions from biomedical text. Given a sentence containing two drug mentions, the model predicts whether the drugs interact and what type of interaction is described — distinguishing between pharmacokinetic mechanisms, clinical effects, medical advice, and generic interactions.

This builds on prior sequence classification (CAD Classification) and token classification (Clinical NER) work by combining both concepts: classifying the *relationship* between entities within a sentence.

## Task Definition

| Label | Meaning | Example |
|-------|---------|---------|
| **mechanism** | Pharmacokinetic mechanism described | "Phenobarbital decreases aspirin effectiveness by enzyme induction" |
| **effect** | Effect or outcome of the interaction | "Aspirin may decrease the effects of probenecid" |
| **advise** | Clinical recommendation or warning | "Aspirin should not be given with antacids" |
| **int** | Interaction stated without details | "Drug X interacts with Drug Y" |
| **negative** | No interaction between the pair | Most pairs (~85% of data) |

## Dataset: DDI Corpus 2013

A SemEval shared task benchmark sourced from two domains:

| Split | Samples | DrugBank | MedLine |
|-------|---------|----------|---------|
| Train | 27,792 | 26,005 | 1,787 |
| Test | 5,716 | 5,265 | 451 |

The dataset is heavily imbalanced — 85.5% of all entity pairs have no interaction. This made **macro F1** the primary evaluation metric rather than accuracy, since a model predicting "negative" for every pair would achieve 85.5% accuracy while being useless.

## Entity Marker Strategy

A single sentence can contain multiple drug pairs, each with a different interaction label. To tell BERT which pair is being classified, entity markers are inserted directly into the text:

**Before:**
```
Milk, milk products, and calcium-rich foods may impair the absorption of EMCYT.
```

**After:**
```
Milk, milk products, and [E1]calcium[/E1]-rich foods may impair the absorption of [E2]EMCYT[/E2].
```

The four markers (`[E1]`, `[/E1]`, `[E2]`, `[/E2]`) are registered as special tokens in BERT's vocabulary so the tokenizer treats each as a single, indivisible token. The embedding matrix is resized from 30,522 to 30,526 to accommodate them, and they learn meaningful representations during fine-tuning. Markers are inserted **right-to-left** (rightmost entity first) to preserve character offsets.

## Model & Training

- **Base model:** `bert-base-uncased` (110M parameters)
- **Classification head:** Linear layer on the `[CLS]` token (768 → 5 classes)
- **Max sequence length:** 256 tokens
- **Learning rate:** 2e-5 with weight decay of 0.01
- **Epochs:** 3 with macro F1 as the model selection metric
- **Device:** Apple M2 (MPS)

## Results

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Macro F1** | **0.79** |
| Weighted F1 | 0.94 |
| Accuracy | 0.94 |

### Per-Class F1 Across Epochs

| Class | Epoch 1 | Epoch 2 | Epoch 3 | Trend |
|-------|---------|---------|---------|-------|
| negative | 0.97 | 0.97 | 0.98 | Stable — dominant class, easy to learn |
| mechanism | 0.71 | 0.80 | 0.80 | Large improvement epoch 1→2, then plateau |
| effect | 0.75 | 0.80 | 0.80 | Improved then stable |
| advise | 0.87 | 0.84 | 0.85 | Consistently strong |
| int | 0.51 | 0.51 | 0.50 | Stuck — insufficient training data (189 samples) |

### Inference Examples

| Input | Prediction | Confidence |
|-------|------------|------------|
| "[E1]Aspirin[/E1] may decrease the effects of [E2]probenecid[/E2]." | effect | 99.80% |
| "[E1]Warfarin[/E1] should not be taken with [E2]aspirin[/E2]." | advise | 99.76% |
| "The patient was prescribed [E1]metformin[/E1] and [E2]lisinopril[/E2]." | negative | 99.97% |

### Key Takeaways

- **`advise` (0.85 F1) was the strongest positive class** — distinctive language patterns like "should not be used with" and "is not recommended" made it easier to classify.
- **`mechanism` and `effect` both reached 0.80 F1** — the two largest positive classes showed a notable jump from epoch 1 to epoch 2 before plateauing.
- **`int` remained stuck at ~0.50 F1** — only 189 training samples (0.7% of data) with a vague definition. This is a data limitation, not a model limitation.
- **Accuracy (0.94) is inflated by the dominant negative class** — macro F1 (0.79) is the more meaningful metric for this task.

## Potential Improvements

- Class weighting or oversampling to boost minority class performance
- Domain-specific pretrained models (BioBERT, PubMedBERT)
- Merging the `int` class into a broader "other interaction" category

## Tools

Python, PyTorch, HuggingFace Transformers, scikit-learn, pandas, HuggingFace Datasets
