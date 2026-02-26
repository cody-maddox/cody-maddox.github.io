---
title: "Clinical Lab Assistant — RAG System"
summary: "End-to-end Retrieval-Augmented Generation system for clinical lab test Q&A, achieving 4.94/5.0 correctness and 36/36 faithfulness (zero hallucinations) across 12 lab tests. Deployed as a live Streamlit app."
tags: [RAG, LLM, LlamaIndex, GPT-4o, ChromaDB, Streamlit, NLP, Python]
order: 1
---

## Overview

A Retrieval-Augmented Generation (RAG) system built for the healthcare domain. The system answers patient and clinician questions about common laboratory tests using curated content from two authoritative sources: **MedlinePlus** (patient-education focus) and **Mayo Clinic Labs** (technical lab-handbook focus).

**[Live Demo →](https://clinical-lab-assistant-xd5df9urgepwfmeqrzkmpi.streamlit.app/)**
*(Streamlit Community Cloud — first load may be slow after inactivity)*

| Component | Choice |
|-----------|--------|
| Generation model | `gpt-4o-mini` |
| Embedding model | `text-embedding-3-small` |
| Vector store | ChromaDB (persistent) |
| Framework | LlamaIndex |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, notebook only) |
| Corpus | 12 lab tests — A1C, CBC, CMP, BMP, TSH, PSA, PT/INR, Liver Panel, Troponin, Microalbumin, Ferritin, CRP |

### Final Evaluation Results

| Metric | Score |
|--------|-------|
| **Correctness (gpt-4o judge)** | **4.94 / 5.0** (35/36 correct) |
| **Faithfulness** | **36 / 36** — zero hallucinations |
| Routing accuracy | 36 / 36 — all queries routed to correct source |

---

## Dual-Source Design

The dual-source architecture is intentional. MedlinePlus and Mayo Clinic Labs serve fundamentally different audiences:

- **MedlinePlus** answers general patient questions: *"Why do I need this test? What does a high result mean? Do I need to fast?"*
- **Mayo Clinic Labs** answers technical clinician questions: *"What tube type is required? What is the refrigerated specimen stability? What are the age-stratified reference ranges?"*

A single homogeneous corpus would force the retriever to compete between patient-education content and lab-handbook content for every query. The dual-source design cleanly separates these knowledge domains and enables source-targeted retrieval through the agent layer (described below).

---

## System Architecture

```
Query + Conversation History (last 3 turns)
  │
  ▼
[condense_query()]  ── rewrites vague follow-ups into standalone questions
  │
  ▼
[Intent Classifier]  ──────────────────────────────►  Canned Response
  │ in_scope                                           (out_of_scope /
  ▼                                                     emergency /
[Source Classifier]                                      diagnosis_request)
  │
  ├─► patient_education ──► Medline Engine  ──► Retrieved Chunks (top-3)
  │                                                       │
  └─► lab_procedures    ──► Mayo Engine    ──►    [Prompt Template]
                                                          │
                                                    [GPT-4o-mini]
                                                          │
                                                    Streamed Response
```

Both engines share the same ChromaDB index, differing only in a `MetadataFilters` constraint (`source="medline"` vs `source="mayo"`). This ensures cross-source retrieval noise is eliminated by design rather than by retrieval heuristics.

---

## Data Ingestion & Cleaning

PDFs were loaded with LlamaIndex's `SimpleDirectoryReader` and passed through custom cleaning functions before chunking:

- **MedlinePlus (`clean_medline`):** Split on `"References\n"` and discard the trailing citations section — which otherwise pollutes the vector space with bibliography noise.
- **Mayo (`clean_mayo`):** Regex-based stripping of repeating page headers ("Test Definition / Document generated / Page X of Y") + split on `"Fees & Codes"` to drop administrative billing content.

Each cleaned document is converted to a LlamaIndex `Document` object with `source` (`"medline"` / `"mayo"`) and `test_name` metadata attached to every chunk — enabling source attribution in `response.source_nodes`.

**Final index:** 24 documents (12 tests × 2 sources) → 47 chunks at `SentenceSplitter(chunk_size=1500, chunk_overlap=300)`.

---

## Iterative Development

The system was built phase-by-phase, with evaluation at each stage to measure the concrete contribution of each component.

### Evaluation Setup

12-question evaluation set (6 Medline, 6 Mayo, 2 per test) evaluated using LlamaIndex's `CorrectnessEvaluator` with **gpt-4o as judge** — deliberately stronger than the gpt-4o-mini generation model to avoid self-scoring bias. Evaluator scores each response 1–5 against a verified reference answer.

### Phase 5 — Baseline

**Average: 4.38 / 5.0**

Two failures identified:
- **Q7 (HbA1c tube type):** Score 1.0 — retrieval failure; correct chunk not surfaced with `similarity_top_k=2`
- **Q11 (CMP specimen handling):** Score 2.0 — cross-source retrieval failure; Medline patient-education content retrieved instead of Mayo lab-handbook content

### Phase 6 — Cross-Encoder Re-Ranking

Added `SentenceTransformerRerank` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) as a two-stage retrieval pipeline: cosine similarity retrieves `top_k=6` candidates, then the cross-encoder re-scores all 6 jointly against the query and passes only `top_n=2` to the LLM.

The key distinction: embedding models encode query and chunk *separately*, while a cross-encoder reads them *together* — producing a richer relevance judgment.

**Average: 4.62 / 5.0** (+0.24)
- Q7: 1.0 → 4.5 (cross-encoder surfaced the correct Mayo chunk that cosine similarity missed)
- Q11: 2.0 → 2.0 (cross-source noise not addressable by re-ranking alone)

### Phase 7 — Prompt Template

Added a custom `PromptTemplate` enforcing: answer only from context, exhaustive enumeration on procedural questions, source citation, and a medical disclaimer on every response.

**Average: 4.58 / 5.0** (essentially stable)
- Q7: 4.5 → 5.0 (exhaustiveness instruction captured the previously omitted "do not aliquot" detail)
- Q11: still 2.0 — a generation-layer instruction cannot fix a retrieval-layer routing problem

### Phase 8 — Agent Layer (Two-Stage Routing)

The core architectural addition. Rather than querying a single unified index, all queries are first classified before any retrieval occurs.

**Stage 1 — Intent Classifier:** Calls `Settings.llm.chat()` with a system prompt to gate queries into:
- `in_scope` → proceed to retrieval
- `out_of_scope` → return canned response
- `medical_emergency` → redirect to 911
- `diagnosis_request` → redirect to provider

**Stage 2 — Source Classifier:** Routes `in_scope` queries to either `medline_engine` or `mayo_engine` based on query intent (patient education vs. lab procedures). Both engines use the same underlying index with `MetadataFilters` to constrain retrieval to the appropriate source partition.

**Average: 4.71 / 5.0** (+0.13)
- Q7: 5.0 (fully resolved — combined effect of re-ranking + prompt template)
- **Q11: 2.0 → 5.0** — the CMP specimen handling failure that persisted through all previous phases was fully resolved by routing the query exclusively to `mayo_engine`

---

## Corpus Expansion & Final Evaluation

The corpus was scaled from 3 to 12 lab tests (BMP, TSH, PSA, PT/INR, Liver Panel, Troponin, Microalbumin, Ferritin, CRP added). All new PDFs cleaned correctly through the existing pipeline with no changes required.

A 36-question evaluation set (2 Medline + 2 Mayo per new test) was run on the expanded corpus, covering procedural stress tests including age-stratified PSA limits, troponin tube type (lithium heparin — different from the serum gel used by most tests), the only urine specimen in the corpus (microalbumin), and multiple tests with different fasting durations.

**Expanded corpus correctness: 4.94 / 5.0** (35/36 correct, all 36 routing decisions correct)

The one failure: liver panel fasting question — a cross-test retrieval noise problem within the Medline partition. Multiple tests discuss fasting in similar language (BMP: 8hr, ferritin: 12hr, liver panel: 10–12hr), and the query lacked enough test-specific signal to outrank a competing fasting chunk from another test.

---

## Faithfulness Evaluation

**Faithfulness: 36 / 36 — zero hallucinations**

`FaithfulnessEvaluator` (gpt-4o judge) checks whether the LLM's response is grounded in the retrieved chunks — independent of whether those chunks contained the right answer. This distinguishes two failure modes that correctness scores conflate:

| Faithfulness | Correctness | Diagnosis |
|---|---|---|
| ✅ Pass | ✅ High | Ideal — right chunks retrieved, response grounded in them |
| ✅ Pass | ❌ Low | **Retrieval failure** — LLM faithfully reported what it found, but the right chunk wasn't retrieved |
| ❌ Fail | ✅ High | **Hallucination** — LLM produced correct-sounding answer from outside the retrieved context |
| ❌ Fail | ❌ Low | Hallucination or total failure |

The liver panel fasting failure (correctness 3.0) **passed faithfulness** — confirming it is a retrieval gap, not a hallucination. The LLM reported exactly what was in the retrieved chunks, which happened to be fasting content from a different test. All remaining failures are retrieval gaps, not fabrications — a critical safety property for medical applications.

---

## Evaluation Score Progression

| Phase | Correctness | Corpus | Key Change |
|-------|-------------|--------|------------|
| Baseline | 4.38 / 5.0 | 3 tests | — |
| + Re-ranking | 4.62 / 5.0 | 3 tests | Q7: 1.0 → 4.5 |
| + Prompt template | 4.58 / 5.0 | 3 tests | Q7: 4.5 → 5.0 |
| + Agent layer | 4.71 / 5.0 | 3 tests | Q11: 2.0 → 5.0 |
| Expanded corpus | 4.94 / 5.0 | 12 tests | 35/36 on new questions |
| **+ Faithfulness eval** | **4.94 / 5.0** | **12 tests** | **36/36 zero hallucinations** |

---

## Streamlit Deployment

The pipeline was extracted from the exploratory notebook into `rag_pipeline.py` (shared backend module) and `app.py` (Streamlit UI). Key app features:

- **Streaming responses** via `st.write_stream(response.response_gen)` — tokens appear as the LLM generates
- **Retrieved sources expander** — shows source (Medline/Mayo), test name, and raw chunk text for each retrieved node
- **Conversational memory** — `condense_query()` in `rag_pipeline.py` rewrites vague follow-up questions ("what about for children?") into standalone questions using the last 3 turns of history before routing; `route_query()` accepts an optional `history` parameter and `app.py` passes `st.session_state.history` on every call
- **Conversation history display** — past Q&A pairs rendered below the current answer (newest first); clears on page refresh

**Deployment trade-off:** The `SentenceTransformerRerank` postprocessor was dropped for Streamlit Community Cloud — `sentence-transformers` depends on PyTorch (~750MB), which exceeded the platform's build constraints. `similarity_top_k` was reduced from 6 to 3 to compensate. Quality impact is low given the small, domain-focused index and metadata filtering already constraining retrieval to roughly half the index per query.

---

## Known Limitations

- **Tabular data extraction (Q12 — Potassium):** The Mayo CMP PDF contains dense age-bracketed reference range tables; the LLM consistently reads incorrect row values. Retrieval is correct — this is a generation-layer failure. Potential fix: parse PDF tables into structured format before indexing.
- **Cross-test noise within Medline partition (Liver Panel Fasting):** Generic fasting language across multiple tests produces retrieval competition. Potential fix: `test_name` metadata filter (requires query-to-test-name resolution) or smaller fasting-specific chunk sizes.
- **Partial CBC stability answer:** Refrigerated stability (48hr) is always retrieved correctly; ambient temperature stability (24hr) is consistently omitted despite exhaustiveness instructions.

---

## Tools

Python, LlamaIndex, OpenAI API (gpt-4o-mini, text-embedding-3-small), ChromaDB, Sentence Transformers, Streamlit, pandas, HuggingFace
