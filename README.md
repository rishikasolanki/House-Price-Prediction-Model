<<<<<<< HEAD
# Week 2: Supervised Learning — Regression and Classification

This folder adds the Week 2 assignments and Mini Project 2. It includes a linear regression house price model, a Titanic logistic regression classifier, and comparisons with decision tree and random forest models.

## House Price Prediction

Download `train.csv` from the Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) competition and place it at `data/train.csv` in this folder. The Kaggle test file does not contain sale prices, so this project creates its own train/test split from the labeled `train.csv` to evaluate R² and MSE.

Run from the repository root:

```bash
python week2_supervised_learning/src/house_price_model.py
```

The script uses `GrLivArea`, `OverallQual`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`, and `FullBath` when present. It imputes missing feature values with medians, trains Linear Regression, Decision Tree, and Random Forest regressors, prints MSE and R², and saves a predicted-versus-actual plot and held-out predictions.

## Titanic Survival Classification

The existing `data/Titanic-Dataset.csv` is used for logistic regression. Run:

```bash
python week2_supervised_learning/src/titanic_classification.py
```

The script splits passengers into training and test sets, fits preprocessing on training data only, then reports accuracy and a confusion matrix. The features are passenger class, sex, age, number of siblings/spouses, number of parents/children, fare, and embarkation port.

## Setup

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The scripts require pandas, matplotlib, and scikit-learn. `train.csv` is not included; download it from Kaggle as described above.
=======
# House-Price-Prediction-Model
A Machine Learning project that predicts house prices using Linear Regression. The model is trained on the Kaggle House Prices dataset and evaluated using R² score. Includes data preprocessing, model training, prediction, performance evaluation, and visualization of predicted vs actual house prices.
>>>>>>> de3d23366f81adefdb03f665025fc52e98e9584b
