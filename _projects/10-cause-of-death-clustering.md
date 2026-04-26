---
title: "Cause of Death Clustering Dashboard"
summary: "Interactive Plotly/Dash dashboard clustering ~2,950 U.S. counties by cause-of-death profiles using K-Means on ~359,000 rows of CDC WONDER data across five snapshot years (2000–2019). Features real-time parameter control, temporal cluster tracking, and demographic tooltips."
tags: [K-Means, Clustering, Data Visualization, Plotly, Dash, CDC WONDER, Public Health, Python, scikit-learn]
order: 3
---

## Overview

An interactive dashboard that groups U.S. counties by their cause-of-death profiles, revealing where people die from similar causes and how those patterns shift over time. Built for **CSE 6242: Data and Visual Analytics** at Georgia Tech (Spring 2026) with a team of six.


| Component | Choice |
|---|---|
| Mortality data | CDC WONDER (~359,000 rows, ~2,950 counties) |
| Demographic data | ACS 5-Year Estimates (2010/2015/2019) + Census Intercensal (2000/2005) |
| Clustering model | K-Means (scikit-learn), k=3–8 |
| Dashboard | Plotly Dash — choropleth map + click-driven info panel |
| Years covered | 2000, 2005, 2010, 2015, 2019 (2020 excluded to avoid COVID-19 outliers) |

---

## Motivation

Geographic disparities in U.S. mortality have persisted for decades, yet no existing tool combines multi-cause clustering with interactive visualization. Static studies examine one or two causes of death in isolation. Interactive tools like the CDC's Heart Disease and Stroke Atlas are limited to a single disease category. This dashboard fills that gap: it clusters counties across 18 ICD-10 cause categories simultaneously and lets users explore those clusters interactively.

---

## Dashboard

<div style="margin: 1.5rem 0;">
  <img src="{{ '/assets/images/dva_dashboard.png' | relative_url }}" alt="Cause of Death Clustering Dashboard — choropleth map of U.S. counties colored by mortality cluster, with county detail panel" style="width: 100%; border-radius: 6px;">
</div>

The dashboard includes:
- **Choropleth map** — counties colored by cluster assignment with descriptive labels (e.g., "Heart Disease Dominant, High Respiratory") rather than generic cluster numbers
- **Filters** — base year (sets cluster centroids), display year (assigns counties via `predict()`), and k (number of clusters, 3–8)
- **Click-driven info panel** — shows demographics, age profile, race breakdown, top causes of death, and 5-year cluster trend for any selected county
- **Suppression-aware tooltips** — instead of excluding counties with suppressed CDC data, suppression counts are surfaced in tooltips so users can judge data reliability

---

## Method

### Data

CDC WONDER provides county-level mortality counts by ICD-10 cause for every year, but suppresses cells with fewer than 10 deaths. This creates a suppression problem: the average county has >50% of its causes suppressed. Excluding suppressed counties would create severe urban bias, excluding most rural areas.

**Solution:** Crude death rate ([Deaths / Population] × 100,000) was computed directly from always-present Deaths and Population columns, bypassing the suppressed Age-Adjusted Rate column. Remaining missing rows (CDC omits rows entirely when deaths < 10) were filled with 0, representing near-zero mortality. This produces a complete feature matrix with 0% NaN.

Of the 120 ICD-10 categories, 38 are aggregate parent categories (prefixed `#`). Of these, **18 had meaningful non-zero rates in ≥10% of counties** and were retained as clustering features.

### Clustering

K-Means was selected for two reasons specific to this use case:
1. Hard cluster assignments map directly onto a choropleth — each county gets one color
2. K-Means' `predict()` method allows new data to be assigned to pre-fit centroids, enabling the **temporal tracking** feature: cluster centroids trained on one base year can be applied to any other year to reveal how counties shift between mortality profiles over time

`StandardScaler` was applied before clustering to prevent high-magnitude causes from dominating Euclidean distance calculations.

All 150 combinations of base year (5) × display year (5) × k (3–8) were pre-computed and stored in `cluster_assignments.csv` for fast dashboard loading.

### Silhouette Analysis

| k | 2000 | 2005 | 2010 | 2015 | 2019 |
|---|---|---|---|---|---|
| 3 | 0.1583 | 0.1690 | 0.1595 | 0.1560 | 0.1508 |
| 4 | 0.1501 | 0.1533 | 0.1440 | 0.1398 | 0.1492 |
| 5 | 0.1525 | 0.1168 | 0.1544 | 0.1477 | 0.1553 |

k=3 is optimal for 4 of 5 analysis years and is used as the dashboard default.

---

## Key Innovations

**1. Interactive clustering with real-time parameter control** — users adjust base year, display year, and k to see updated county groupings instantly. Existing literature uses static visualizations with fixed parameters.

**2. Temporal cluster tracking** — select a base year to lock cluster centroids, then change the display year to see how counties migrate between mortality profiles. Enables questions like "which counties now resemble the high-risk profile from 2000?"

**3. Confidence-aware suppression handling** — rather than excluding suppressed counties (which would drop most rural areas), suppression counts per county are exposed as a tooltip, letting users make informed judgments about data reliability.

**4. Multi-cause feature representation** — counties are clustered on a vector of 18 cause-specific death rates simultaneously, rather than the single-cause analyses common in prior work.

---

## Evaluation

User testing (n=8) measured confidence before and after using the dashboard via Likert scale:

| Question | Before | After |
|---|---|---|
| Confidence grouping counties by cause of death | 2.5/5 | 4.0/5 |
| Confidence identifying county-level trends over time | 2.5/5 | 4.5/5 |

Usability ratings during testing:

| Task | Rating |
|---|---|
| Ability to use tooltip features | 4.9/5 |
| Ability to adjust year of overlay | 4.8/5 |
| Ability to adjust year of analysis | 4.5/5 |
| Ability to use filters | 4.5/5 |
| Ability to open the dashboard | 4.4/5 |

---

## Limitations

- **Crude rates vs. age-adjusted rates:** Older-skewing counties may show artificially higher rates independent of true disease burden. Age-adjusted rates were not used because CDC suppresses them when deaths ≤ 20, which affected 37.4% of county-cause pairs.
- **Suppression filling:** Treating suppressed cells as zero cannot be verified against true counts and may underestimate rare causes in rural counties.
- **K-Means assumptions:** K-Means assumes spherical, equally-sized clusters — an assumption that does not fully hold for mortality data, as evidenced by consistently low silhouette scores across all years. Alternative methods (hierarchical clustering, DBSCAN) could be explored to validate robustness.

---

## Tools

Python, Plotly Dash, scikit-learn, pandas, NumPy, CDC WONDER, U.S. Census Bureau (ACS 5-Year Estimates, Intercensal Datasets)

*Team: Amy Burton, AJ Druck, Kendall Lightcap, Cody Maddox, Pankaj Padwal, Charlie Schmitter — CSE 6242, Georgia Tech, Spring 2026*
