"""Train and evaluate house-price regression models on Kaggle train.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF", "YearBuilt", "FullBath"]
TARGET = "SalePrice"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Download train.csv from Kaggle House Prices "
            "and place it in the project's data folder."
        )

    df = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    results = {}
    for name, estimator in models.items():
        model = make_pipeline(SimpleImputer(strategy="median"), estimator)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = predictions
        print(f"{name}: MSE={mean_squared_error(y_test, predictions):,.2f}, "
              f"R²={r2_score(y_test, predictions):.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ActualSalePrice": y_test, "PredictedSalePrice": results["Linear Regression"]},
                 index=y_test.index).sort_index().to_csv(OUTPUT_DIR / "linear_regression_predictions.csv")

    actual = y_test.to_numpy()
    predicted = results["Linear Regression"]
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.65, color="teal", edgecolors="none")
    plt.plot([low, high], [low, high], "r--", linewidth=1.5, label="Perfect prediction")
    plt.title("House Prices: Predicted vs Actual (Linear Regression)")
    plt.xlabel("Actual sale price ($)")
    plt.ylabel("Predicted sale price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=160)
    plt.close()
    print(f"Saved predictions and plot to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
