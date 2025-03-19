# Traffic-accident-prediction-XAI
A comprehensive framework for traffic accident prediction using Wasserstein GAN with gradient penalty for imbalanced data generation, XGBoost for prediction with temporal features, and SHAP for model interpretation.
## Overview

This repository presents a novel framework for traffic accident prediction that addresses three critical challenges in transportation safety modeling:
1. The severe class imbalance inherent in accident data
2. The temporal heterogeneity of accident occurrence patterns
3. The limited interpretability of traditional prediction models

Our approach combines advanced generative modeling for data balancing, sophisticated feature engineering for temporal pattern capture, and state-of-the-art model interpretation techniques to create a comprehensive and practical accident prediction system.

## Background

Traffic accidents remain a significant public safety concern worldwide, with substantial human and economic costs. Accurate prediction of accident occurrence can facilitate proactive traffic management and resource allocation, potentially saving lives and reducing congestion. However, traffic accident prediction faces several challenges:

- Accident events are rare compared to normal traffic conditions, leading to highly imbalanced datasets
- Accident patterns vary significantly across different time periods (holidays, peak hours, etc.)
- Complex, black-box prediction models offer limited insights for practical implementation of safety measures

This research addresses these challenges through a multi-faceted approach combining data augmentation, temporal-aware modeling, and interpretable machine learning.

## Methodology

### Data Imbalance Handling

We implement a novel data generation approach using Wasserstein Generative Adversarial Networks (WGAN) with gradient penalty terms. This approach:
- Generates high-quality synthetic accident samples to balance the dataset
- Avoids mode collapse issues common in traditional GANs
- Preserves the statistical properties of real accident data

### Temporal Feature Engineering

Our framework incorporates specialized time classification features that capture:
- Time-of-day effects
- Holiday vs. non-holiday patterns
- Peak hour vs. off-peak characteristics

These features enable the model to account for temporal heterogeneity in accident risk patterns.

### Prediction Model

We employ an enhanced Extreme Gradient Boosting (XGBoost) model for accident prediction, which offers:
- Superior handling of complex, non-linear relationships in traffic data
- Effective integration of diverse feature types
- Robust performance on imbalanced data (even after rebalancing)
- Inherent feature importance ranking

### Model Interpretation

We apply SHapley Additive exPlanations (SHAP) to interpret the model results, which provides:
- Global feature importance rankings
- Local explanations for individual predictions
- Insights into non-linear relationships between features and accident risk
- Visual representation of complex interactions between factors


## Keywords

Traffic safety, Accident prediction, Imbalanced data, Generative adversarial networks, XGBoost, Interpretable machine learning, SHAP values, Temporal heterogeneity, Transportation engineering
