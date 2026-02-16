---
title: "Obesity Classification & Sampling Artifact Analysis"
summary: "Trained and tuned multiple ML classifiers on a semi-synthetic obesity dataset, then used subgroup analysis to detect sampling artifacts introduced by SMOTE oversampling — revealing inflated gender-obesity associations."
tags: [Classification, Random Forest, XGBoost, Fairness, Python]
order: 6
---

## Overview

This project trains multiple machine learning classifiers to predict obesity levels from demographic and behavioral survey data. But beyond model accuracy, the central question is: **what happens when 77% of your dataset is synthetically generated?**

The dataset — from the UCI Machine Learning Repository — was heavily augmented using SMOTE to balance underrepresented obesity classes. While this improves class balance, it can introduce artificial feature-label relationships that don't exist in the real population. This project develops a tuned Random Forest classifier, then uses subgroup analysis to identify where synthetic oversampling has distorted the data.

## Dataset

- **Source:** UCI ML Repository — *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*
- **Samples:** 2,111 records from residents of Mexico, Peru, and Colombia via web survey
- **Features:** 16 demographic, lifestyle, and behavioral attributes (age, gender, dietary habits, physical activity, transportation, etc.)
- **Target:** 7 obesity categories — Insufficient Weight, Normal Weight, Overweight I/II, Obesity Type I/II/III
- **Key caveat:** Only 23% of records are real survey responses; 77% were synthetically generated via SMOTE using the Weka tool

## Model Development

### Baseline Screening

Six models were trained on the preprocessed data to screen candidates for further tuning. BMI was removed prior to modeling to avoid target leakage.

| Model | Baseline Accuracy |
|-------|-------------------|
| Dummy Classifier | 0.142 |
| Logistic Regression | 0.624 |
| Linear SVM | 0.681 |
| RBF SVM | 0.740 |
| Random Forest | 0.877 |
| XGBoost | 0.863 |

*Table 2 from the report. Linear models underperform due to the nonlinear structure of the data. The two ensemble methods clearly separate from the rest, so only RF and XGBoost were selected for hyperparameter tuning.*

### Tuned Ensemble Performance

Both models were tuned via 5-fold cross-validation with grid search over hyperparameters.

| Model | CV Accuracy | CV Macro Recall | CV Macro F1 |
|-------|-------------|-----------------|-------------|
| Random Forest | 0.865 | 0.861 | 0.863 |
| XGBoost | 0.861 | 0.858 | 0.860 |

*Table 3. RF slightly outperformed XGBoost across all three metrics. Combined with its greater interpretability (feature importances), RF was selected as the global model.*

### Global Model — Test Set Performance

| Model | Accuracy | Macro Recall | Macro F1 |
|-------|----------|--------------|----------|
| Random Forest | 0.861 | 0.860 | 0.862 |

*Table 6. Test performance closely matches CV performance, indicating the model generalizes well and is not overfit.*

### Confusion Matrix

![RF confusion matrix]({{ '/assets/images/RF_confusion_matrix.png' | relative_url }})

*Most misclassifications occur between adjacent obesity categories (e.g., Obesity Type I vs II), which reflects the ordinal structure of the task. Severe misclassifications are rare.*

### Feature Importance

![RF feature importance]({{ '/assets/images/RF_importances.png' | relative_url }})

*Age, FCVC (vegetable consumption), height, and CH2O (water intake) are the top predictors — consistent with established obesity research. However, `Gender_Male` and `CAEC_Sometimes` rank higher than expected, a signal explored further below.*

## Sampling Artifact Detection

### Gender Subgroup Performance

| Gender | Accuracy | Macro Recall | n |
|--------|----------|--------------|---|
| Female | 0.886 | 0.741 | 201 |
| Male | 0.838 | 0.704 | 222 |

*Table 7. Despite similar sample sizes, the model performs notably better on female samples — 4.8 percentage points higher in accuracy and 3.7 points in macro recall.*

**Why this is suspicious:** Real-world obesity research shows that obesity prevalence does not differ significantly by gender after accounting for lifestyle factors (Cooper et al., 2021). Yet the gender countplot from EDA reveals that several obesity categories are nearly absent from one gender while overrepresented in the other:

![Gender distribution]({{ '/assets/images/gender_countplot.png' | relative_url }})

*Obesity Type II is almost entirely absent from females; Obesity Type III is heavily skewed toward females. These stark separations are unlikely to reflect real population patterns.*

This suggests SMOTE created cleaner feature-label boundaries for female samples, inflating apparent model performance on that subgroup. The model isn't better at classifying females because of genuine signal — it's exploiting synthetic structure that wouldn't exist in real data.

### Mutual Information Corroborates the Artifact

`Gender_Male` ranked 6th in mutual information with the target (MI = 0.231), higher than family history with overweight and most behavioral features. This level of predictive power for gender is not supported by obesity literature and further indicates that oversampling artificially strengthened the gender-obesity association.

## Key Takeaways

- **High accuracy doesn't mean the model is trustworthy** — the RF achieved 86% accuracy, but subgroup analysis reveals the performance is partially built on synthetic artifacts
- **SMOTE can introduce hidden biases** — it improved class balance but distorted feature-label relationships, particularly for gender and snacking frequency (CAEC)
- **Subgroup analysis is essential** for any dataset with synthetic augmentation — global metrics alone can mask these distortions
- Interpretations of model behavior should be **restricted to the semi-synthetic context** and not generalized to real-world obesity risk

## Tools & Libraries

Python, scikit-learn, XGBoost, pandas, matplotlib, seaborn
