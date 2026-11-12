"""Train and evaluate house price models using the included Housing.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Housing.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TARGET = "price"
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
CATEGORICAL_FEATURES = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus",
]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find the included housing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Housing.csv is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Housing.csv contains no rows")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = pd.to_numeric(df[TARGET], errors="raise")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])
    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    predictions = {}
    for name, estimator in estimators.items():
        model = make_pipeline(preprocessing, estimator)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        print(f"{name}: MSE={mean_squared_error(y_test, pred):,.2f}, "
              f"R²={r2_score(y_test, pred):.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ActualPrice": y_test, "PredictedPrice": predictions["Linear Regression"]},
                 index=y_test.index).sort_index().to_csv(OUTPUT_DIR / "linear_regression_predictions.csv")

    actual = y_test.to_numpy()
    predicted = predictions["Linear Regression"]
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.65, color="teal", edgecolors="none")
    plt.plot([low, high], [low, high], "r--", linewidth=1.5, label="Perfect prediction")
    plt.title("Housing Dataset: Predicted vs Actual Price")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=160)
    plt.close()
    print(f"Saved predictions and plot to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
