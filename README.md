# EMBER 2024 MAMBA Multi-Output Malware Behavior Analyser

This project applies **machine learning techniques** to the **multi-output, multi-label classification** task presented
by the **EMBER 2024 dataset (Behavior subset)**.  
The goal is to identify and predict multiple behavioral traits of malware samples simultaneously using classical ML
algorithms adapted for multi-output learning.  
It was developed by **Alessio Murgioni** and **Riccardo Deidda** as the **final project for the Machine Learning course
**.

## Overview

The **EMBER 2024** dataset, originally created for malcious/benign classification, also provides behavioral features
extracted from malware.  
This project focuses exclusively on the **behavioral subset**, which encodes malware categories such as ransomwares,
backdoors, etc...

Each sample may exhibit **multiple behaviors** at once — making this a **multi-output, multi-label classification**
problem rather than a standard single-label one.

## Implemented Models

The following machine learning models are implemented and compared:

- **K-Nearest Neighbors (KNN)**
- **Random Forest (RF)**
- **Gradient Boosting (GB)**

Since Random Forest and Gradient Boosting are not inherently multi-output, they are wrapped with **Scikit-learn
meta-estimators** to extend their functionality:

- **`OneVsRestClassifier`** — trains a separate classifier for each label.
- **`MultiOutputClassifier`** — fits one model per output, allowing correlated target predictions.

This design allows direct comparison between **base learners** and **multi-output strategies**.

Each model is evaluated using the following performance metrics:

| Metric           | Description                                                      |
|:-----------------|:-----------------------------------------------------------------|
| **precision**    | Proportion of correctly predicted positive labels                |
| **recall**       | Ability to retrieve all relevant labels                          |
| **f1_macro**     | F1-score averaged equally across all labels                      |
| **f1_micro**     | F1-score weighted by label frequency                             |
| **hamm_loss**    | Fraction of incorrectly predicted labels (Hamming loss)          |
| **class_report** | Detailed classification report per label (precision, recall, F1) |
| **conf_matrix**  | Confusion matrix representing true vs. predicted labels          |

These metrics collectively evaluate all implemented models performances.
