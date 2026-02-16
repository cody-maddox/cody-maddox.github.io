---
title: "Drug Review Sentiment Analysis"
summary: "Fine-tuned BERT for binary sentiment classification on patient drug reviews, achieving 0.80 macro F1 on multi-field review text with moderate class imbalance."
tags: [NLP, BERT, Sentiment Analysis, Biomedical, Python]
order: 4
---

## Overview

Fine-tuned BERT for binary sentiment classification on patient drug reviews. Given a patient's review text covering benefits, side effects, and general comments about a medication, the model predicts whether the overall sentiment is positive or negative.

This project applies sequence classification to a new domain (drug review sentiment) using the same BERT fine-tuning pipeline established in prior projects (CAD Classification, DDI Relation Extraction), with the added challenge of working with multi-field text inputs and a smaller dataset.

## Dataset

**DrugLib.com Drug Review Dataset** from the UCI Machine Learning Repository. Each review is a structured patient report on a specific medication containing three text fields: **benefitsReview**, **sideEffectsReview**, and **commentsReview**, along with a satisfaction rating (1-10).

| Split | Total | Positive (7-10) | Negative (1-6) | % Positive |
|-------|-------|-----------------|----------------|------------|
| Train | 3,107 | 2,130 | 977 | 68.6% |
| Test | 1,036 | 670 | 366 | 64.7% |

Ratings were binned into binary labels — **positive (7-10)** and **negative (1-6)**. The cutoff at 6 was chosen because a 6/10 on a drug review represents a lukewarm experience rather than a genuine endorsement, and it helps balance the class distribution compared to a cutoff at 5.

The rating distribution is right-skewed — most patients rate medications highly, with a secondary spike at rating 1 (strongly negative experiences).

## Approach

### Text Preprocessing

All three review fields were concatenated into a single input to give the model the fullest context of the patient's experience — benefits alone or side effects alone could be misleading. Null values in review columns (up to 75 missing in sideEffectsReview) were filled with empty strings before concatenation.

### Model & Training

- **Base model:** `bert-base-uncased` (110M parameters)
- **Classification head:** Linear layer on the `[CLS]` token (768 → 2 classes)
- **Max sequence length:** 512 (BERT's maximum — mean review length ~120 words, some reaching 875)
- **Learning rate:** 2e-5 with weight decay of 0.01
- **Batch size:** 8 (reduced from 16 due to MPS memory constraints at max_length=512)
- **Epochs:** 3 with macro F1 as the model selection metric
- **Device:** Apple M2 (MPS)

## Results

### Overall Metrics (Best — Epoch 3)

| Metric | Score |
|--------|-------|
| **Macro F1** | **0.80** |
| Weighted F1 | 0.82 |
| Accuracy | 0.82 |

### Per-Class Performance Across Epochs

| Class | Epoch 1 P/R/F1 | Epoch 2 P/R/F1 | Epoch 3 P/R/F1 |
|-------|----------------|----------------|----------------|
| Negative | 0.75 / 0.67 / 0.71 | 0.83 / 0.54 / 0.65 | 0.80 / 0.66 / 0.73 |
| Positive | 0.83 / 0.88 / 0.85 | 0.79 / 0.94 / 0.86 | 0.83 / 0.91 / 0.87 |
| **Macro F1** | **0.78** | **0.75** | **0.80** |

### Inference Examples

| Review | Prediction | Confidence |
|--------|------------|------------|
| "This medication completely changed my life. My symptoms are gone and I feel amazing. No side effects at all." | positive | 99.20% |
| "Terrible drug. Made me dizzy, nauseous, and unable to sleep. Did nothing for my condition." | negative | 99.65% |
| "It helped a little with my pain but the side effects were rough. Not sure if it's worth continuing." | negative | 97.80% |

### Key Takeaways

- **Macro F1 of 0.80** is a strong result for only 3,107 training samples with moderate class imbalance (68/32 split).
- **Epoch 2 showed an interesting pattern** — negative recall dropped to 0.54 (model became biased toward positive) while positive recall spiked to 0.94. Epoch 3 recovered balance, suggesting the model needed additional training to stabilize.
- **Positive class consistently outperformed negative** (0.87 vs 0.73 F1) — expected given the ~2:1 class imbalance.
- **Negative recall (0.66) is the main weakness** — the model misses about a third of negative reviews. This is the typical failure mode with imbalanced binary classification.
- **Ambiguous reviews lean negative** — the model learned that hedging language ("helped a little", "not sure if it's worth continuing") signals negative sentiment, aligning with the 1-6 rating bin.

## Potential Improvements

- Class weighting or oversampling to improve negative class recall
- Dynamic padding instead of fixed max_length=512 for more efficient memory usage
- Domain-specific pretrained model (BioBERT, PubMedBERT)
- Three-way classification (negative/neutral/positive) with a larger dataset

## Tools

Python, PyTorch, HuggingFace Transformers, scikit-learn, pandas, HuggingFace Datasets
