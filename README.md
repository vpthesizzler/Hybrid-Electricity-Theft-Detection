# Hybrid Machine Learning-Based Electricity Theft Detection in Smart Grids Using Load and Frequency Features

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Random%20Forest-orange)
![Smart Grid](https://img.shields.io/badge/Application-Smart%20Grid-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Research](https://img.shields.io/badge/Project-Research%20Implementation-success)

---

## Overview

Electricity theft is a major challenge for modern power systems, causing revenue losses, operational inefficiencies, and reliability concerns for distribution networks.

This repository presents a **hybrid machine learning framework for electricity theft detection in smart grids** by integrating consumer load behaviour features with power system frequency information.

The proposed approach combines **Random Forest (RF)** and **Extreme Gradient Boosting (XGBoost)** classifiers through a weighted soft-voting ensemble to improve detection capability and robustness under imbalanced smart meter data conditions.

The framework demonstrates the potential of artificial intelligence-based methods for enhancing security, monitoring, and resilience in future smart grid infrastructures.

---

## Research Contributions

The main contributions of this work include:

- Development of a hybrid RF-XGBoost machine learning detection framework.
- Integration of smart meter load characteristics with frequency-based grid features.
- Feature engineering pipeline for electricity theft classification.
- Handling of class imbalance using data balancing techniques.
- Comparative evaluation of individual and ensemble machine learning models.
- Development of a reproducible Python-based research workflow.

---

# Methodology

The proposed framework follows the workflow below:

```
![Workflow Diagram](figures/workflow.png)


---

# Machine Learning Framework

The implemented models include:

## Random Forest

A tree-based ensemble learning algorithm used for robust classification and feature importance analysis.

## XGBoost

An optimized gradient boosting algorithm designed for high-performance classification tasks.

## Hybrid Ensemble Model

The final detection model combines Random Forest and XGBoost predictions using weighted soft voting:

```
Hybrid Prediction =
Weighted combination of RF probability + XGBoost probability
```

---

# Repository Structure

```
hybrid-electricity-theft-detection/

│
├── data/
│ └── README.md
│
├── notebooks/
│ └── Electricity_Theft_Detection.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── train_model.py
│ └── evaluate.py
│
├── figures/
│ ├── workflow.png
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ └── feature_importance.png
│
├── results/
│ ├── metrics.csv
│ └── classification_report.txt
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# Dataset

This research uses smart grid electricity consumption data combined with frequency-related grid information.

Due to dataset licensing restrictions, the original datasets are **not included in this repository**.

Users should obtain the required datasets from their respective official sources and place them inside the `data/` directory.

A sample dataset structure is provided for demonstration purposes.

---

# Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/hybrid-electricity-theft-detection.git
```

Navigate to the project folder:

```bash
cd hybrid-electricity-theft-detection
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

The project was developed using:

```
Python >= 3.10

numpy
pandas
scikit-learn
xgboost
imbalanced-learn
matplotlib
joblib
```

---

# Results

The proposed framework is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Example comparison:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | - | - | - | - | - |
| XGBoost | - | - | - | - | - |
| Hybrid Ensemble | - | - | - | - | - |

*(Results will be updated with final publication values.)*

---

# Visual Results

The repository contains:

- Model workflow diagram
- Confusion matrix
- ROC curve
- Feature importance analysis
- Performance comparison plots

---

# Future Extensions

Future development directions include:

- Explainable Artificial Intelligence (XAI) using SHAP.
- Federated Learning-based privacy-preserving detection.
- Digital Twin integration for smart grid resilience.
- Real-time electricity theft monitoring systems.

---

# Publication

**Title:**

Hybrid Machine Learning-Based Electricity Theft Detection in Smart Grids Using Load and Frequency Features

**Author:**

Vrushabh Patil

**Status:**

Under Review

**Citation:**

```
Vrushabh Patil,
"Hybrid Machine Learning-Based Electricity Theft Detection in Smart Grids Using Load and Frequency Features",
Under Review.
```

---

# Author

## Vrushabh Patil

MSc New and Renewable Energy
Durham University

Research Interests:

- Smart Grid Security
- Artificial Intelligence for Energy Systems
- Machine Learning Applications
- Electricity Theft Detection
- Renewable Energy Integration

---

# Acknowledgements

This work acknowledges the contribution of open-source Python machine learning libraries:

- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib

---

# License

This project is licensed under the MIT License.

---

# Contact

For research collaboration or technical discussions, please contact:

**Vrushabh Patil**
