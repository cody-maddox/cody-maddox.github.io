---
title: "Drug-Drug Interaction Extraction"
summary: "Relation extraction system using BERT with entity markers to classify drug-drug interactions from the DDI Corpus 2013, handling 5 interaction types."
tags: [NLP, BERT, Relation Extraction, Biomedical, Python]
order: 3
---

## Overview

Drug-drug interactions (DDIs) are a major cause of adverse drug events. Manually curating DDI databases is slow and incomplete. This project automates the extraction of DDI relations from biomedical text using a BERT-based classification approach with entity markers.

Given a sentence containing two drug mentions, the model classifies the relationship into one of five categories.

## Task Definition

| Label | Meaning | Example |
|-------|---------|---------|
| **mechanism** | Pharmacokinetic mechanism described | "Phenobarbital decreases aspirin effectiveness by enzyme induction" |
| **effect** | Effect or outcome of the interaction | "Aspirin may decrease the effects of probenecid" |
| **advise** | Clinical recommendation or warning | "Aspirin should not be given with antacids" |
| **int** | Interaction stated without details | "Drug X interacts with Drug Y" |
| **negative** | No interaction between the pair | Most pairs (~85% of data) |

## Dataset: DDI Corpus 2013

- **Sources:** DrugBank drug descriptions + MedLine biomedical abstracts
- **Format:** XML files with sentence text, entity annotations, and pairwise relation labels
- **Training:** 27,792 samples (26,005 DrugBank + 1,787 MedLine)
- **Test:** 5,716 samples (5,265 DrugBank + 451 MedLine)
- **Class imbalance:** 85.5% of training pairs are negative (no interaction)

## Pipeline

### 1. XML Parsing

Each XML file contains sentences with entity and pair annotations. The parser extracts one sample per entity pair — building an entity lookup by ID, then extracting every `<pair>` element with its relation label.

### 2. Entity Marker Insertion

To tell BERT which two drugs we're asking about, markers are inserted directly into the sentence text:

**Before:**
```
Milk, milk products, and calcium-rich foods may impair the absorption of EMCYT.
```

**After:**
```
Milk, milk products, and [E1]calcium[/E1]-rich foods may impair the absorption of [E2]EMCYT[/E2].
```

Markers are inserted **right-to-left** (rightmost entity first) so that character offsets remain valid after insertion.

### 3. Special Token Registration

The four markers (`[E1]`, `[/E1]`, `[E2]`, `[/E2]`) are registered as special tokens in the tokenizer. This ensures they are treated as single, indivisible tokens rather than being split into characters. The model's embedding layer is resized to accommodate the new vocabulary entries.

### 4. Classification

The marked sentences are tokenized, padded to a max length of 256, and fed through BERT for sequence classification into the 5 relation types.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Entity markers over entity embeddings | Simpler approach; directly encodes entity position in the input text |
| Right-to-left insertion | Avoids offset invalidation when inserting markers |
| First span only for multi-span entities | Covers the vast majority of cases; simplifies offset handling |
| Special token registration | Prevents markers from being tokenized as individual characters |

## Tools & Libraries

Python, PyTorch, Hugging Face Transformers, xml.etree.ElementTree, pandas
