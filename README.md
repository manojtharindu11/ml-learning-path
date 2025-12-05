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

1. Linear Regression (Single Variable)

  - Demonstrates simple linear regression with a single variable using home prices and area data.
  - Algorithm: Linear Regression (`sklearn.linear_model.LinearRegression`)

2. Linear Regression (Multiple Variables)

  - Shows linear regression with multiple variables (features) to predict home prices.
  - Algorithm: Linear Regression (`sklearn.linear_model.LinearRegression`)

3. Gradient Descent

  - Explains the gradient descent optimization algorithm, applied to test scores data.
  - Algorithm: Custom Gradient Descent (manual implementation)

4. Save Model Using Joblib and Pickle

  - Teaches how to save and load machine learning models using Joblib and Pickle.
  - Algorithm: Model Serialization (`pickle`, `joblib`)

17. L1 and L2 Regularization (Lasso and Ridge Regression)
  - Reduces overfitting and underfitting in regression models.
  - Algorithm: L1 Regularization (Lasso), L2 Regularization (Ridge)

### Data Preprocessing

5. Dummy Variables and One-Hot Encoding

  - Covers encoding categorical variables using dummy variables and one-hot encoding.
  - Algorithm: One-Hot Encoding, Dummy Variables (`pandas.get_dummies`)

6. Training and Testing Data
  - Focuses on splitting data into training and testing sets, using car prices as an example.
  - Algorithm: Data Splitting (`sklearn.model_selection.train_test_split`)

### Classification - Binary & Multiclass

7. Logistic Regression (Binary Classification)

  - Introduces logistic regression for binary classification, e.g., predicting insurance purchase.
  - Algorithm: Logistic Regression (`sklearn.linear_model.LogisticRegression`)

8. Logistic Regression (Multiclass Classification)
  - Explains multiclass classification using logistic regression, with digit recognition as an example.
  - Algorithm: Logistic Regression (`sklearn.linear_model.LogisticRegression`)

### Tree-Based Algorithms

9. Decision Tree

  - Demonstrates decision tree classification, e.g., predicting high salary based on features.
  - Algorithm: Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`)

11. Random Forest Algorithm

  - Shows how to use the Random Forest algorithm for classification, e.g., digit recognition.
  - Algorithm: Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)

20. Bagging
  - Demonstrates ensemble methods using bagging for improved model performance.
  - Algorithm: Bagging Classifier (`sklearn.ensemble.BaggingClassifier`)

### Distance & Kernel-Based Algorithms

10. Support Vector Machine (SVM)

  - Introduces Support Vector Machine (SVM) for classification, using the iris dataset.
  - Algorithm: Support Vector Machine (`sklearn.svm.SVC`)

18. K-Nearest Neighbors Classification
  - Demonstrates the KNN algorithm for classification tasks.
  - Algorithm: K-Nearest Neighbors (`sklearn.neighbors.KNeighborsClassifier`)

### Probabilistic Algorithms

14. Naive Bayes Classifier (Part 1)

  - Introduces the Naive Bayes classifier, applied to the Titanic dataset.
  - Algorithm: Gaussian Naive Bayes (`sklearn.naive_bayes.GaussianNB`)

15. Naive Bayes Classifier (Part 2)
  - Continues with Naive Bayes, focusing on spam detection in messages.
  - Algorithm: Multinomial Naive Bayes (`sklearn.naive_bayes.MultinomialNB`)

### Unsupervised Learning

13. K-Means Clustering Algorithm

  - Demonstrates K-means clustering for unsupervised learning, e.g., clustering income data.
  - Algorithm: K-Means Clustering (`sklearn.cluster.KMeans`)

19. Principal Component Analysis
  - Introduces dimensionality reduction using PCA for feature extraction.
  - Algorithm: Principal Component Analysis (`sklearn.decomposition.PCA`)

### Model Evaluation & Optimization

12. K-Fold Cross Validation

  - Explains K-fold cross-validation for robust model evaluation, using multiple classifiers.
  - Algorithm: K-Fold Cross Validation (`sklearn.model_selection.cross_val_score`) with Logistic Regression, SVM, and Random Forest

16. Hyperparameter Tuning (GridSearchCV)
  - Demonstrates hyperparameter tuning to find optimal model parameters.
  - Algorithm: Grid Search CV (`sklearn.model_selection.GridSearchCV`), Randomized Search CV (`sklearn.model_selection.RandomizedSearchCV`)

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

- Regression: Linear
- Classification: Logistic Regression, SVM, Decision Trees, Random Forests, Naive Bayes, KNN
- Clustering: K-Means
- Dimensionality Reduction: PCA
- Model Optimization: Hyperparameter tuning, Cross-validation, Regularization
- Data Processing: One-hot encoding, Train-test splitting
- Model Serialization: Joblib, Pickle

## Acknowledgement

This repository was created following machine learning tutorials to build a comprehensive learning resource.
