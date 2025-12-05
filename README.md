# Machine Learning Learning Path

A comprehensive collection of machine learning tutorials and exercises covering fundamental to advanced algorithms using scikit-learn, pandas, and numpy.

## Overview

This repository contains 20+ tutorials demonstrating various machine learning concepts, from regression and classification to clustering and dimensionality reduction. Each tutorial includes Jupyter notebooks with explanations, code examples, and practical exercises.

## Prerequisites

- Python 3.7+
- Required packages are listed in `requirements.txt`

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Tutorials

### Regression

- **Tutorial 0001** - Linear Regression (Single Variable)

  - Demonstrates simple linear regression with a single variable using home prices and area data.
  - **Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)

- **Tutorial 0002** - Linear Regression (Multiple Variables)

  - Shows linear regression with multiple variables (features) to predict home prices.
  - **Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)

- **Tutorial 0003** - Gradient Descent

  - Explains the gradient descent optimization algorithm, applied to test scores data.
  - **Algorithm:** Custom Gradient Descent (manual implementation)

- **Tutorial 0004** - Save Model Using Joblib and Pickle

  - Teaches how to save and load machine learning models using Joblib and Pickle.
  - **Algorithm:** Model Serialization (`pickle`, `joblib`)

- **Tutorial 0017** - L1 and L2 Regularization (Lasso and Ridge Regression)
  - Reduces overfitting and underfitting in regression models.
  - **Algorithm:** L1 Regularization (Lasso), L2 Regularization (Ridge)

### Data Preprocessing

- **Tutorial 0005** - Dummy Variables and One-Hot Encoding

  - Covers encoding categorical variables using dummy variables and one-hot encoding.
  - **Algorithm:** One-Hot Encoding, Dummy Variables (`pandas.get_dummies`)

- **Tutorial 0006** - Training and Testing Data
  - Focuses on splitting data into training and testing sets, using car prices as an example.
  - **Algorithm:** Data Splitting (`sklearn.model_selection.train_test_split`)

### Classification - Binary & Multiclass

- **Tutorial 0007** - Logistic Regression (Binary Classification)

  - Introduces logistic regression for binary classification, e.g., predicting insurance purchase.
  - **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)

- **Tutorial 0008** - Logistic Regression (Multiclass Classification)
  - Explains multiclass classification using logistic regression, with digit recognition as an example.
  - **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)

### Tree-Based Algorithms

- **Tutorial 0009** - Decision Tree

  - Demonstrates decision tree classification, e.g., predicting high salary based on features.
  - **Algorithm:** Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`)

- **Tutorial 0011** - Random Forest Algorithm

  - Shows how to use the Random Forest algorithm for classification, e.g., digit recognition.
  - **Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)

- **Tutorial 0020** - Bagging
  - Demonstrates ensemble methods using bagging for improved model performance.
  - **Algorithm:** Bagging Classifier (`sklearn.ensemble.BaggingClassifier`)

### Distance & Kernel-Based Algorithms

- **Tutorial 0010** - Support Vector Machine (SVM)

  - Introduces Support Vector Machine (SVM) for classification, using the iris dataset.
  - **Algorithm:** Support Vector Machine (`sklearn.svm.SVC`)

- **Tutorial 0018** - K-Nearest Neighbors Classification
  - Demonstrates the KNN algorithm for classification tasks.
  - **Algorithm:** K-Nearest Neighbors (`sklearn.neighbors.KNeighborsClassifier`)

### Probabilistic Algorithms

- **Tutorial 0014** - Naive Bayes Classifier (Part 1)

  - Introduces the Naive Bayes classifier, applied to the Titanic dataset.
  - **Algorithm:** Gaussian Naive Bayes (`sklearn.naive_bayes.GaussianNB`)

- **Tutorial 0015** - Naive Bayes Classifier (Part 2)
  - Continues with Naive Bayes, focusing on spam detection in messages.
  - **Algorithm:** Multinomial Naive Bayes (`sklearn.naive_bayes.MultinomialNB`)

### Unsupervised Learning

- **Tutorial 0013** - K-Means Clustering Algorithm

  - Demonstrates K-means clustering for unsupervised learning, e.g., clustering income data.
  - **Algorithm:** K-Means Clustering (`sklearn.cluster.KMeans`)

- **Tutorial 0019** - Principal Component Analysis
  - Introduces dimensionality reduction using PCA for feature extraction.
  - **Algorithm:** Principal Component Analysis (`sklearn.decomposition.PCA`)

### Model Evaluation & Optimization

- **Tutorial 0012** - K-Fold Cross Validation

  - Explains K-fold cross-validation for robust model evaluation, using multiple classifiers.
  - **Algorithm:** K-Fold Cross Validation (`sklearn.model_selection.cross_val_score`) with Logistic Regression, SVM, and Random Forest

- **Tutorial 0016** - Hyperparameter Tuning (GridSearchCV)
  - Demonstrates hyperparameter tuning to find optimal model parameters.
  - **Algorithm:** Grid Search CV (`sklearn.model_selection.GridSearchCV`), Randomized Search CV (`sklearn.model_selection.RandomizedSearchCV`)

## Structure

Each tutorial folder contains:

- `*.ipynb` - Main tutorial notebook with explanations and code examples
- `exercise.ipynb` - Practice exercises (where available)
- `csv/` - Datasets used in the tutorial (where applicable)
- `outputs/` - Generated outputs and results (where applicable)

## Running the Tutorials

1. Navigate to a tutorial folder
2. Open the Jupyter notebook: `jupyter notebook <notebook_name>.ipynb`
3. Follow along with the explanations and run the code cells
4. Complete the exercises to solidify your understanding

## Topics Covered

- **Regression**: Linear, Ridge, Lasso
- **Classification**: Logistic Regression, SVM, Decision Trees, Random Forests, Naive Bayes, KNN
- **Clustering**: K-Means
- **Dimensionality Reduction**: PCA
- **Model Optimization**: Hyperparameter tuning, Cross-validation, Regularization
- **Data Processing**: One-hot encoding, Train-test splitting
- **Model Serialization**: Joblib, Pickle

## Acknowledgement

This repository was created following machine learning tutorials to build a comprehensive learning resource.