"""
train_regression.py
--------------------
Trains and compares regression models to predict `package_lpa` (salary).

KEY DESIGN DECISIONS:

1. ONLY placed students are used (`placement_status == 'Placed'`). This
   isn't optional - Not Placed students have no real salary (package_lpa is
   NaN for them, by definition, see Phase 2/3), so including them would
   mean training on fabricated/invalid target values.

2. `placement_status` itself is EXCLUDED from the feature set. Once we've
   filtered to placed students, this column is constant ("Placed" for every
   row) and carries zero information - including it would be meaningless,
   not leakage exactly, but pointless noise/a constant column that some
   encoders would mishandle.

3. Train-test split happens before any preprocessing is fit - same
   leakage-safe Pipeline approach as Phase 6.

4. We again compare WITH and WITHOUT the Phase 5 engineered features.

5. We report MAE, MSE, RMSE, and R² - not just R² - because MAE/RMSE are in
   the same units as salary (LPA) and are directly interpretable ("the
   model is off by about X LPA on average"), which matters a lot more to a
   non-technical stakeholder than an abstract R² value.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

RAW_NUMERIC = [
    "cgpa", "iq_score", "tenth_percentage", "twelfth_percentage",
    "graduation_percentage", "internships", "projects",
    "technical_skills_score", "communication_score", "backlogs",
    "certifications", "work_experience_months",
]
ENGINEERED_NUMERIC = [
    "academic_score", "skill_score", "experience_score",
    "overall_performance_score",
]
CATEGORICAL = ["branch"]
TARGET = "package_lpa"


def build_pipeline(model, numeric_features):
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(random_state=RANDOM_STATE),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def run_comparison(df: pd.DataFrame, numeric_features: list, label: str):
    placed = df[df["placement_status"] == "Placed"].copy()
    X = placed[numeric_features + CATEGORICAL]
    y = placed[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []
    fitted_pipelines = {}
    for name, model in get_models().items():
        pipe = build_pipeline(model, numeric_features)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        metrics["Model"] = name
        results.append(metrics)
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(results).set_index("Model")[["MAE", "MSE", "RMSE", "R2"]].round(4)
    print(f"\n=== Regression comparison ({label}) - n={len(placed)} placed students ===")
    print(results_df.sort_values("RMSE"))

    return results_df, fitted_pipelines, (X_test, y_test)


if __name__ == "__main__":
    df = pd.read_csv("data/placement_data_featured.csv")

    results_raw, pipelines_raw, test_raw = run_comparison(df, RAW_NUMERIC, "raw features only")
    results_eng, pipelines_eng, test_eng = run_comparison(
        df, RAW_NUMERIC + ENGINEERED_NUMERIC, "raw + engineered features"
    )

    results_raw.to_csv("reports/regression_results_raw.csv")
    results_eng.to_csv("reports/regression_results_engineered.csv")

    print("\nSaved comparison tables to reports/")
