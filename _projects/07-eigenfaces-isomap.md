---
title: "Eigenfaces & ISOMAP Dimensionality Reduction"
summary: "PCA-based eigenface decomposition for facial recognition combined with ISOMAP manifold learning, including systematic epsilon tuning and comparison against PCA baselines."
tags: [PCA, ISOMAP, Dimensionality Reduction, Computer Vision, Python]
order: 7
---

## Overview

Dimensionality reduction is essential for working with high-dimensional image data. This project explores two complementary approaches: **Eigenfaces** (linear, PCA-based) for facial recognition and **ISOMAP** (nonlinear, manifold-based) for discovering the underlying structure of a face image dataset. Both are applied to the Yale Faces dataset.

## Part 1: Eigenfaces

### The Idea

Every face image can be represented as a point in a high-dimensional pixel space. PCA finds the directions of maximum variance in that space — the **eigenfaces** — which capture the most important visual patterns across a set of face images. A new face can then be projected onto these eigenfaces, and the **projection residual** tells us how well that face is explained by a given subject's eigenface basis.

### Pipeline

1. **Preprocessing** — Load Yale face images (GIF format), convert to grayscale, normalize to [0,1], downsample by 4x, and flatten each image into a vector
2. **PCA** — Fit PCA on each subject's training images (excluding test images) to extract the top 6 eigenfaces
3. **Projection residuals** — Project each subject's test image onto both subjects' eigenface bases using only PC1, and compute the squared L2 residual:

$$s_{ij} = \| z_i - U_j U_j^T z_i \|_2^2$$

where $z_i$ is the mean-centered test image and $U_j$ is the top eigenvector for subject $j$.

### Eigenface Visualizations

#### Subject 1 — Top 3 Eigenfaces
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
  <img src="{{ '/assets/images/subject1_eigenface_01.png' | relative_url }}" alt="Subject 1 Eigenface 1">
  <img src="{{ '/assets/images/subject1_eigenface_02.png' | relative_url }}" alt="Subject 1 Eigenface 2">
  <img src="{{ '/assets/images/subject1_eigenface_03.png' | relative_url }}" alt="Subject 1 Eigenface 3">
</div>

#### Subject 2 — Top 3 Eigenfaces
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
  <img src="{{ '/assets/images/subject2_eigenface_01.png' | relative_url }}" alt="Subject 2 Eigenface 1">
  <img src="{{ '/assets/images/subject2_eigenface_02.png' | relative_url }}" alt="Subject 2 Eigenface 2">
  <img src="{{ '/assets/images/subject2_eigenface_03.png' | relative_url }}" alt="Subject 2 Eigenface 3">
</div>

*Each eigenface captures a different mode of variation — lighting direction, expression, etc.*

### Projection Residuals

| Residual | Value | Interpretation |
|----------|-------|----------------|
| s11 (Subject 1 test on Subject 1 basis) | 102.30 | Low — good match |
| s12 (Subject 2 test on Subject 1 basis) | 605.51 | High — poor match |
| s21 (Subject 1 test on Subject 2 basis) | 501.31 | High — poor match |
| s22 (Subject 2 test on Subject 2 basis) | 56.02 | Low — good match |

The diagonal residuals (s11, s22) are much smaller than the off-diagonal ones — confirming that eigenfaces can distinguish between subjects. A test image produces a small residual when projected onto its own subject's basis and a large residual on a different subject's basis.

---

## Part 2: ISOMAP

### The Idea

Unlike PCA, which finds linear projections, ISOMAP discovers the **nonlinear manifold** that the data lies on. It does this by:

1. Building a nearest-neighbor graph based on Euclidean distances within an epsilon neighborhood
2. Computing shortest-path (geodesic) distances between all pairs of points
3. Applying classical MDS (multidimensional scaling) to embed the geodesic distance matrix into a low-dimensional space

### Epsilon Tuning

The epsilon parameter controls graph connectivity. Too small and the graph fragments; too large and it connects distant points, destroying manifold structure. Systematic tuning from epsilon = 10.0 to 13.25 revealed the transition from a fragmented graph to a well-connected "Mickey Mouse" structure.

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
  <img src="{{ '/assets/images/nn_graph_eps_10.50.png' | relative_url }}" alt="NN Graph epsilon=10.50">
  <img src="{{ '/assets/images/nn_graph_eps_12.00.png' | relative_url }}" alt="NN Graph epsilon=12.00">
  <img src="{{ '/assets/images/nn_graph_eps_12.75.png' | relative_url }}" alt="NN Graph epsilon=12.75">
</div>

*Left to right: epsilon = 10.50 (fragmented), 12.00 (connecting), 12.75 (optimal — clear cluster structure).*

### ISOMAP Algorithm Implementation

With the optimal epsilon = 12.75:

1. **Adjacency matrix** — Connect points within epsilon using Euclidean distance; use 0 for non-neighbors
2. **Geodesic distances** — Compute all-pairs shortest paths via `scipy.sparse.csgraph.shortest_path`
3. **Double centering** — Apply the centering matrix $H = I - \frac{1}{m}\mathbf{1}\mathbf{1}^T$ to get the Gram matrix: $G = -\frac{1}{2} H D^2 H$
4. **Eigendecomposition** — Extract the top 2 eigenvectors of $G$, scaled by $\sqrt{\lambda}$

The resulting 2D embedding was visualized as a scatterplot with face image annotations, alongside a PCA baseline for comparison.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 4x downsampling | Reduces dimensionality while preserving enough facial structure for PCA |
| Only PC1 for residuals | Single principal component is sufficient to discriminate between two subjects |
| Mean-centering test images | Aligns test images with the training distribution before projection |
| Sparse adjacency matrix | Efficient shortest-path computation on large graphs via scipy |
| Epsilon = 12.75 | Visually confirmed full connectivity with clear manifold structure |

## Tools & Libraries

Python, NumPy, scikit-learn (PCA), SciPy (cdist, shortest_path), NetworkX, matplotlib, Pillow
