# Machine Learning Learning Path

A compact set of Jupyter notebooks that walk through core machine learning concepts with scikit-learn, pandas, and numpy. Topics span regression, classification, clustering, dimensionality reduction, and model evaluation.

## Table of Contents

- Overview
- Quick Start
- Tutorial Index
- Folder Structure
- Topics Covered
- Acknowledgement

## Overview

20+ short tutorials with runnable notebooks and small datasets. Each entry focuses on one idea with minimal setup and clear code examples.

## Quick Start

Requirements: Python 3.7+ and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
jupyter notebook
```

Open any tutorial notebook and run the cells in order.

## Tutorial Index

### Regression

1. Linear Regression (Single Variable) — home prices vs. area; algorithm: Linear Regression
2. Linear Regression (Multiple Variables) — multi-feature home prices; algorithm: Linear Regression
3. Gradient Descent — optimizing a simple cost function; algorithm: manual gradient descent
4. Save Model Using Joblib and Pickle — persist and load models; tools: pickle, joblib
5. L1 and L2 Regularization — Lasso and Ridge to reduce overfitting; algorithms: L1, L2 regularization

### Data Preprocessing

5. Dummy Variables and One-Hot Encoding — encode categorical features; tools: pandas.get_dummies
6. Training and Testing Data — split car price data; tool: train_test_split

### Classification

7. Logistic Regression (Binary) — insurance purchase prediction; algorithm: Logistic Regression
8. Logistic Regression (Multiclass) — digit recognition; algorithm: Logistic Regression
9. Decision Tree — salary prediction classification; algorithm: DecisionTreeClassifier
10. Support Vector Machine — iris dataset; algorithm: SVC
11. Random Forest — digit recognition; algorithm: RandomForestClassifier
12. K-Nearest Neighbors — basic KNN classification; algorithm: KNeighborsClassifier
13. Bagging — ensemble with bagging; algorithm: BaggingClassifier

### Probabilistic Methods

14. Naive Bayes Part 1 — Titanic survival; algorithm: GaussianNB
15. Naive Bayes Part 2 — spam detection; algorithm: MultinomialNB

### Unsupervised Learning

13. K-Means Clustering — income clustering; algorithm: KMeans
14. Principal Component Analysis — feature reduction; algorithm: PCA

### Evaluation and Tuning

12. K-Fold Cross Validation — compare models with cross_val_score
13. Hyperparameter Tuning — grid and random search with GridSearchCV and RandomizedSearchCV

## Folder Structure

- `*/` tutorial folders containing:
  - `*.ipynb` notebook(s) with the walkthrough
  - `exercise.ipynb` optional practice (where present)
  - `csv/` sample datasets (where present)
  - `outputs/` generated results (where present)

## Topics Covered

- Regression: Linear, L1, L2
- Classification: Logistic, SVM, Decision Trees, Random Forests, Naive Bayes, KNN, Bagging
- Clustering: K-Means
- Dimensionality Reduction: PCA
- Model Evaluation: Cross-validation
- Hyperparameter Tuning: Grid search, random search
- Data Prep: One-hot encoding, train/test split
- Model Persistence: pickle, joblib

## Acknowledgement

This repository was created following machine learning tutorials to build a concise learning resource.
