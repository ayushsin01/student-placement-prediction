"""
hyperparameter_tuning.py
-------------------------
Tunes the strongest candidate models from Phase 6/7 using GridSearchCV.

WHAT ARE HYPERPARAMETERS?
Settings chosen BEFORE training that control how a model learns (e.g. how
strong Lasso's regularization is, how many trees a Random Forest builds,
how deep a tree can grow) - as opposed to parameters (like linear
regression's coefficients), which the model learns FROM the data.

WHY THEY MATTER:
Phase 7 showed Lasso Regression scored R^2 ~ 0 (useless) with its DEFAULT
alpha=1.0 - not because Lasso is a bad fit for this problem, but because
the default regularization strength was too aggressive for this data's
scale, zeroing out too many useful coefficients. This is a direct,
concrete demonstration of why hyperparameters matter: the same algorithm
can look "broken" or "excellent" purely based on hyperparameter choice.

APPROACH:
- GridSearchCV with 5-fold cross-validation, so each hyperparameter
  combination is validated on 5 different train/validation splits of the
  TRAINING data only (test set stays completely untouched until final
  evaluation) - this keeps model SELECTION leakage-free too, not just
  preprocessing.
- Small, deliberately bounded grids (per Section 11: avoid excessive
  computation) - we're searching a few sensible values around defaults,
  not thousands of combinations.
- Classification models scored on F1 (matches Phase 6's primary metric).
- Regression models scored on negative RMSE (sklearn's convention -
  GridSearchCV always maximizes, so we minimize RMSE by maximizing its
  negative).
"""

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestRegressor,
                               GradientBoostingRegressor)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, mean_absolute_error,
                              mean_squared_error, r2_score)

RANDOM_STATE = 42

RAW_NUMERIC = [
    "cgpa", "iq_score", "tenth_percentage", "twelfth_percentage",
    "graduation_percentage", "internships", "projects",
    "technical_skills_score", "communication_score", "backlogs",
    "certifications", "work_experience_months",
]
ENGINEERED_NUMERIC = RAW_NUMERIC + [
    "academic_score", "skill_score", "experience_score",
    "overall_performance_score",
]
CATEGORICAL = ["branch"]


def build_pipeline(model, numeric_features):
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def tune_classification(df):
    X = df[RAW_NUMERIC + CATEGORICAL]
    y = df["placement_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}

    # --- Logistic Regression (raw features - Phase 6 winner on ROC-AUC) ---
    pipe = build_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), RAW_NUMERIC)
    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs"],
    }
    grid = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, list(grid.classes_).index("Placed")]
    results["Logistic Regression"] = {
        "best_params": grid.best_params_,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred, pos_label="Placed"),
        "test_roc_auc": roc_auc_score((y_test == "Placed").astype(int), y_proba),
    }

    # --- Gradient Boosting (engineered features - Phase 6 winner on F1) ---
    X2 = df[ENGINEERED_NUMERIC + CATEGORICAL]
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pipe = build_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE), ENGINEERED_NUMERIC)
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.05, 0.1],
    }
    grid = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=5, n_jobs=-1)
    grid.fit(X2_train, y2_train)
    y_pred = grid.predict(X2_test)
    y_proba = grid.predict_proba(X2_test)[:, list(grid.classes_).index("Placed")]
    results["Gradient Boosting"] = {
        "best_params": grid.best_params_,
        "test_accuracy": accuracy_score(y2_test, y_pred),
        "test_f1": f1_score(y2_test, y_pred, pos_label="Placed"),
        "test_roc_auc": roc_auc_score((y2_test == "Placed").astype(int), y_proba),
    }

    return results


def tune_regression(df):
    placed = df[df["placement_status"] == "Placed"].copy()
    X = placed[RAW_NUMERIC + CATEGORICAL]
    y = placed["package_lpa"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    results = {}

    def eval_reg(grid, X_test, y_test):
        y_pred = grid.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return {
            "best_params": grid.best_params_,
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_rmse": float(np.sqrt(mse)),
            "test_r2": r2_score(y_test, y_pred),
        }

    # --- Ridge ---
    pipe = build_pipeline(Ridge(random_state=RANDOM_STATE), RAW_NUMERIC)
    grid = GridSearchCV(pipe, {"model__alpha": [0.01, 0.1, 1, 10, 50, 100]},
                         scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    results["Ridge Regression"] = eval_reg(grid, X_test, y_test)

    # --- Lasso (this is the one we expect to improve a lot) ---
    pipe = build_pipeline(Lasso(random_state=RANDOM_STATE, max_iter=5000), RAW_NUMERIC)
    grid = GridSearchCV(pipe, {"model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1]},
                         scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    results["Lasso Regression"] = eval_reg(grid, X_test, y_test)

    # --- Random Forest ---
    pipe = build_pipeline(RandomForestRegressor(random_state=RANDOM_STATE), RAW_NUMERIC)
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_leaf": [1, 3, 5],
    }
    grid = GridSearchCV(pipe, param_grid, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    results["Random Forest Regressor"] = eval_reg(grid, X_test, y_test)

    # --- Gradient Boosting ---
    pipe = build_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE), RAW_NUMERIC)
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.05, 0.1],
    }
    grid = GridSearchCV(pipe, param_grid, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    results["Gradient Boosting Regressor"] = eval_reg(grid, X_test, y_test)

    return results


if __name__ == "__main__":
    df = pd.read_csv("data/placement_data_featured.csv")

    print("Tuning classification models...")
    clf_results = tune_classification(df)
    for name, res in clf_results.items():
        print(f"\n{name}: {res}")

    print("\n\nTuning regression models...")
    reg_results = tune_regression(df)
    for name, res in reg_results.items():
        print(f"\n{name}: {res}")

    with open("reports/tuning_results.json", "w") as f:
        json.dump({"classification": clf_results, "regression": reg_results}, f, indent=2, default=str)

    print("\nSaved reports/tuning_results.json")
