---
title: "Obesity Level Classification"
summary: "Machine learning pipeline classifying obesity levels from lifestyle and physical attributes using Random Forest and other classifiers, with feature importance analysis."
tags: [Classification, Random Forest, EDA, Python]
order: 5
---

## Overview

Obesity is a major global health concern linked to diabetes, cardiovascular disease, and other conditions. This project builds a multi-class classification pipeline to predict obesity levels from survey data covering eating habits, physical activity, and demographic features.

## Dataset

- **Source:** UCI Machine Learning Repository — Estimation of Obesity Levels
- **Samples:** Survey responses with 16 features covering age, gender, dietary habits, physical activity, transportation, and more
- **Target:** 7 obesity level categories ranging from underweight to obesity type III
- **Includes both real and synthetically-generated samples** to increase dataset size

## Exploratory Data Analysis

### Feature Correlation
![Correlation heatmap]({{ '/assets/images/corr_heatmap.png' | relative_url }})

### Gender Distribution
![Gender distribution]({{ '/assets/images/gender_countplot.png' | relative_url }})

### Food Consumption Between Meals
![CAEC distribution]({{ '/assets/images/CAEC_countplot.png' | relative_url }})

## Model & Results

### Random Forest Classifier

The Random Forest model achieved the best performance across all obesity categories.

#### Confusion Matrix
![RF confusion matrix]({{ '/assets/images/RF_confusion_matrix.png' | relative_url }})

#### Feature Importance
![RF feature importance]({{ '/assets/images/RF_importances.png' | relative_url }})

The most predictive features were **weight**, **age**, and **dietary habits** — aligning with medical understanding of obesity risk factors.

## Tools & Libraries

Python, scikit-learn, pandas, matplotlib, seaborn
