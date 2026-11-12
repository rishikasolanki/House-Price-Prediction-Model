# Week 2: Supervised Learning

This project uses the included `data/Housing.csv` dataset to train and compare regression models that predict house prices.

## House price prediction

Run from this folder:

```bash
python src/house_price_model.py
```

The script uses `price` as the target, with the numeric property fields and home amenity fields as features. Missing values are imputed, categorical values are one-hot encoded, and Linear Regression, Decision Tree, and Random Forest models are evaluated with mean squared error (MSE) and R². It saves held-out linear regression predictions and a predicted-versus-actual plot under `outputs/`.

Install the required packages with:

```bash
python -m pip install pandas matplotlib scikit-learn
```

## Titanic classification exercise

`src/titanic_classification.py` is a separate exercise and requires a Titanic passenger dataset with the columns listed in that script. It is not required to run the included housing project.
