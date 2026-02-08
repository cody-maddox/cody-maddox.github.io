---
title: "MRI Image Reconstruction via Compressed Sensing"
summary: "Reconstructing a 50x50 MRI image from incomplete, noisy measurements using Lasso and Ridge regression with cross-validated regularization tuning."
tags: [Compressed Sensing, Lasso, Ridge, Medical Imaging, Python]
order: 8
---

## Overview

In medical imaging, acquiring fewer measurements means faster scan times and lower patient exposure. **Compressed sensing** makes this possible by reconstructing images from far fewer measurements than the original image dimensions — provided the image is sparse in some basis.

This project reconstructs a 50x50 MRI image (2,500 pixels) from only 1,300 noisy linear measurements using regularized regression.

## Problem Setup

The measurement model is:

$$y = Ax + \varepsilon$$

where:
- **x** is the vectorized 50x50 MRI image (2,500 dimensions)
- **A** is a 1,300 x 2,500 Gaussian random measurement matrix
- **epsilon** is Gaussian noise with standard deviation 5
- **y** is the observed measurement vector (1,300 dimensions)

This is an **underdetermined system** — there are more unknowns (2,500) than measurements (1,300). Without regularization, there are infinitely many solutions. Regularization introduces a preference for solutions that are sparse (Lasso) or small-norm (Ridge).

### Original MRI Image

![Original MRI]({{ '/assets/images/mri_original.png' | relative_url }})

## Lasso Regression (L1 Regularization)

Lasso promotes **sparsity** — it drives many pixel values to exactly zero, which is appropriate when the image has a sparse representation. This aligns with the core assumption of compressed sensing.

- **Method:** `LassoCV` with 10-fold cross-validation over an automatically selected alpha grid
- **Optimal lambda:** 0.1104

### Lasso CV Error Curve

![Lasso CV curve]({{ '/assets/images/mri_lasso_cv_curve.png' | relative_url }})

*The red dashed line marks the optimal regularization strength. Too little regularization overfits to noise; too much destroys image detail.*

### Lasso Reconstructed Image

![Lasso reconstruction]({{ '/assets/images/mri_lasso_reconstructed.png' | relative_url }})

## Ridge Regression (L2 Regularization)

Ridge penalizes the **sum of squared coefficients** rather than driving them to zero. This produces a smoother reconstruction but doesn't enforce sparsity.

- **Method:** `GridSearchCV` with Ridge regression over 81 alpha values (log-spaced from 10 to 1,000), 10-fold CV
- **Optimal lambda:** 199.53

### Ridge CV Error Curve

![Ridge CV curve]({{ '/assets/images/mri_ridge_cv_curve.png' | relative_url }})

### Ridge Reconstructed Image

![Ridge reconstruction]({{ '/assets/images/mri_ridge_reconstructed.png' | relative_url }})

## Comparison

| Method | Optimal Lambda | Behavior |
|--------|---------------|----------|
| **Lasso** | 0.1104 | Sparse solution — sharper edges, some artifacts from zeroed-out pixels |
| **Ridge** | 199.53 | Smooth solution — softer overall, distributes energy across all pixels |

Lasso produces a reconstruction more aligned with compressed sensing theory, where sparsity is the key prior. Ridge produces a smoother but less sharp reconstruction because it shrinks all coefficients toward zero rather than eliminating them.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Gaussian random measurement matrix | Satisfies the Restricted Isometry Property (RIP) with high probability |
| `fit_intercept=False` | The measurement model has no intercept term |
| Log-spaced alpha grid for Ridge | Covers a wide range of regularization strengths efficiently |
| 10-fold CV for both methods | Robust estimate of generalization error |
| LassoCV auto alpha selection | Lets scikit-learn find the optimal search range |

## Tools & Libraries

Python, NumPy, SciPy (loadmat), scikit-learn (LassoCV, Ridge, GridSearchCV), matplotlib
