---
title: "K-Means Image Compression"
summary: "Custom K-Means clustering implementation applied to image compression, comparing L1 and L2 loss functions across different K values to analyze the quality-compression tradeoff."
tags: [Clustering, K-Means, Image Processing, Python]
order: 7
---

## Overview

Image compression is a natural application of K-Means clustering. By replacing each pixel's color with its nearest cluster centroid, we can represent an image using only K colors instead of the full color spectrum. This project implements K-Means from scratch and explores how the number of clusters (K) and the choice of loss function (L1 vs L2) affect compression quality.

## Approach

### Custom K-Means Implementation

Rather than using scikit-learn's built-in KMeans, this project implements the algorithm from scratch:

1. **Initialize** K centroids randomly from the data points
2. **Assign** each pixel to its nearest centroid (using L1 or L2 distance)
3. **Update** centroids as the mean (L2) or median (L1) of assigned pixels
4. **Repeat** until convergence

### Loss Functions

- **L2 (Euclidean)** — Standard K-Means; minimizes squared distances. Centroids are computed as the **mean** of assigned points. More sensitive to outliers.
- **L1 (Manhattan)** — K-Medians variant; minimizes absolute distances. Centroids are computed as the **median**. More robust to outlier colors.

## Results

Images were compressed at K = 3, 6, 12, 24, and 48 clusters using both loss functions.

### Parrots — L2 Loss
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
  <img src="{{ '/assets/images/parrots_K3_L2.png' | relative_url }}" alt="K=3">
  <img src="{{ '/assets/images/parrots_K12_L2.png' | relative_url }}" alt="K=12">
  <img src="{{ '/assets/images/parrots_K48_L2.png' | relative_url }}" alt="K=48">
</div>

*Left to right: K=3, K=12, K=48. More clusters preserve finer color detail.*

### Parrots — L1 Loss
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
  <img src="{{ '/assets/images/parrots_K3_L1.png' | relative_url }}" alt="K=3">
  <img src="{{ '/assets/images/parrots_K12_L1.png' | relative_url }}" alt="K=12">
  <img src="{{ '/assets/images/parrots_K48_L1.png' | relative_url }}" alt="K=48">
</div>

*L1 loss produces subtly different color palettes — median centroids tend to select more "typical" colors.*

### Football Network — L2 Loss
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
  <img src="{{ '/assets/images/football_K3_L2.png' | relative_url }}" alt="K=3">
  <img src="{{ '/assets/images/football_K12_L2.png' | relative_url }}" alt="K=12">
  <img src="{{ '/assets/images/football_K48_L2.png' | relative_url }}" alt="K=48">
</div>

## Key Observations

- **K=3** produces a striking posterization effect — images are reduced to just 3 colors
- **K=12** is surprisingly good — most of the visual quality is recovered with relatively few colors
- **K=48** is nearly indistinguishable from the original for most images
- **L1 vs L2** differences are subtle but visible in areas with smooth gradients — L1 tends to produce slightly more saturated centroids
- The quality-compression tradeoff is non-linear: the jump from K=3 to K=12 is far more impactful than K=24 to K=48

## Tools & Libraries

Python, NumPy, Pillow, matplotlib
