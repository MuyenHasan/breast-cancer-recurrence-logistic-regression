# Breast Cancer Recurrence Prediction with Logistic Regression

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/breast-cancer-recurrence-logistic-regression/blob/main/breast_cancer_recurrence_logistic_regression.ipynb)

Course project for *AI Expert System and Machine Learning*. A logistic regression classifier that predicts whether breast cancer recurs, covering the full workflow from data checks to overfitting analysis.

## Dataset
UCI Machine Learning Repository, Breast Cancer dataset: 286 patients, 9 categorical features, about 30% recurrence. The notebook loads the data from a public copy of the CSV file.

## What the notebook covers
- Data checks: missing values, duplicates, class imbalance
- Preprocessing in one scikit-learn Pipeline: stratified train/test split first (no data leakage), most-frequent imputation, ordinal encoding for ordered bands, one-hot encoding for unordered categories, scaling
- Baseline (always predicts the majority class) versus logistic regression
- Evaluation: confusion matrix, precision, recall, F1, ROC curve and AUC, precision-recall curve
- Overfitting and underfitting: train vs test, 5-fold cross-validation, learning curve, validation curve, and a deliberate underfit / good fit / overfit experiment
- Improvement: GridSearchCV for C, class weights for imbalance
- Interpretation: odds ratios, and a prediction for a new patient

## Results (hold-out test set, 58 patients)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy baseline (always "no recurrence") | 0.707 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression (default) | 0.655 | 0.286 | 0.118 | 0.167 | 0.604 |
| Tuned C = 0.01 | 0.707 | 0.000 | 0.000 | 0.000 | 0.669 |
| Tuned C + balanced class weights (final) | 0.586 | 0.348 | 0.471 | 0.400 | 0.671 |

5-fold cross-validated ROC-AUC on the training data: 0.686 (default) and 0.714 (tuned).

## Key takeaways
- Accuracy is misleading on imbalanced data: a model that always says "no recurrence" has higher accuracy than the default model but finds no recurrences.
- Stronger regularisation reduced overfitting (train/validation AUC gap fell from about 0.09 to 0.04).
- Class weights raised recall from 0 to 0.47, at the cost of more false alarms.
- Tumor grade, number of affected lymph nodes and node capsule involvement push the prediction towards recurrence (associations, not causes).

## Limitations
Small dataset (only 58 test patients), coarse feature bands, class imbalance, 9 imputed values, a linear model, and no external validation. **This is a learning project, not a medical tool.**

## How to run
Click the Colab badge above and choose **Runtime > Run all**. To run locally: `pip install numpy pandas matplotlib seaborn scikit-learn jupyter`, then open the notebook with Jupyter.

## Reference
UCI Machine Learning Repository, Breast Cancer dataset (Institute of Oncology, University Medical Centre Ljubljana).
