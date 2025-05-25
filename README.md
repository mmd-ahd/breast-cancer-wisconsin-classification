# Breast Cancer Classification Using Machine Learning 

This project focuses on building and evaluating machine learning models to classify breast cancer tumors as benign (0) or malignant (1) based on the Breast Cancer Wisconsin (Diagnostic) Dataset.

## Introduction

Breast cancer is a significant health concern globally. This project leverages various machine learning techniques to develop robust predictive models. The goal is to accurately classify tumors, which can be a crucial step in early diagnosis and treatment planning. The models are built using 30 real-valued features from 569 instances in the dataset.

---

## Dataset Overview

* **Source:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Samples:** 569 instances
* **Features:** 30 real-valued measurements
* **Target:** Diagnosis (Benign = 0, Malignant = 1)
* **File:** `data.csv` (as loaded in the notebook)

---

## Project Outline

The project follows a structured machine learning workflow:

1.  **Data Loading & Initial Preprocessing:** Importing necessary libraries, loading the dataset, and performing initial cleaning (e.g., dropping uninformative columns like 'id' and 'Unnamed: 32', encoding the target variable).
2.  **Exploratory Data Analysis (EDA):** Understanding class distribution, descriptive statistics, and feature correlations using visualizations like count plots and heatmaps.
3.  **Data Preprocessing Pipeline:** Checking for missing values and scaling features using `StandardScaler`.
4.  **Feature Selection & Dimension Reduction:**
    * **Univariate Feature Selection:** `SelectKBest` with `f_classif`.
    * **Recursive Feature Elimination (RFE):** Using `LogisticRegression` as the estimator.
    * **Feature Importance:** From `RandomForestClassifier`.
    * **Principal Component Analysis (PCA):** For dimensionality reduction.
    * The project selected 8 features for the final models. RFE-selected features were used for hyperparameter tuning and final evaluation.
5.  **Model Building & Cross Validation:**
    * A variety of models were evaluated using 5-fold stratified cross-validation:
        * Logistic Regression
        * Linear Discriminant Analysis (LDA)
        * Decision Tree
        * Random Forest
        * K-Nearest Neighbors (KNN)
        * Gaussian Naive Bayes
        * Support Vector Machine (SVM)
        * Ensemble (Voting Classifier with Logistic Regression, Random Forest, and SVM)
    * Metrics used for evaluation included Accuracy, Precision, Recall, F1-score, and ROC AUC.
6.  **Hyperparameter Tuning:**
    * `GridSearchCV` was used for Logistic Regression, SVM, and the Ensemble model, optimizing for ROC AUC.
    * Tuning was performed on RFE-selected features.
7.  **Final Evaluation on Test Data:** The tuned models (Logistic Regression, SVM, Ensemble) were evaluated on the held-out test set.
8.  **Model Interpretation:**
    * Feature importances from the tuned Logistic Regression model.
    * Permutation feature importance for the tuned SVM model.
    * Analysis of ensemble component contributions.
9.  **Conclusions & Future Work:** Summarizing findings and suggesting potential next steps.

---

## Technologies & Libraries Used

* **Python 3**
* **Core Libraries:**
    * NumPy
    * Pandas
    * Matplotlib
    * Seaborn
* **Scikit-learn:**
    * `train_test_split`
    * `Pipeline`
    * `StandardScaler`
    * `SelectKBest`, `f_classif`
    * `RFE`
    * `PCA`
    * `LogisticRegression`
    * `DecisionTreeClassifier`
    * `RandomForestClassifier`
    * `VotingClassifier`
    * `KNeighborsClassifier`
    * `GaussianNB`
    * `SVC` (Support Vector Classifier)
    * `LinearDiscriminantAnalysis`
    * `cross_validate`, `GridSearchCV`, `StratifiedKFold`
    * `classification_report`, `roc_curve`, `auc`
    * `permutation_importance`

