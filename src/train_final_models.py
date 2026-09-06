"""
train_final_models.py
----------------------
Locks in the final selected models from Phase 9, reports their honest
held-out test performance, then retrains on the FULL dataset and saves the
complete pipelines (preprocessing + model together) for deployment.

FINAL MODEL CHOICES (from Phase 9 evidence):
- Placement classification: Logistic Regression, C=0.01, raw features.
  Chosen over Gradient Boosting because it has the best ROC-AUC (0.789 vs
  0.779) AND is simpler/more interpretable (coefficients are directly
  readable), which matters for a system making decisions about people.
- Salary regression: Ridge Regression, alpha=10, raw features.
  Chosen over Lasso (near-identical R^2, 0.591 vs 0.590) because Ridge
  keeps all features with small shrinkage rather than potentially zeroing
  some out - slightly more stable for a small feature set like this one,
  and ties out with Linear Regression's original Phase 7 result almost
  exactly, giving extra confidence this is a genuine, non-fluke result.

WHY WE RETRAIN ON THE FULL DATASET BEFORE SAVING:
Phase 6-9 all used an 80/20 train-test split so we could get an HONEST,
held-out estimate of how well each model generalizes - that 20% test set
must never influence training, or the reported metrics become optimistic
and untrustworthy. That process is now complete and its numbers (reported
below) are the true, defensible performance figures for the README/resume.

Once model SELECTION is finalized, it's standard practice to retrain the
chosen model on 100% of the available data (train + test combined) before
shipping it - more training data generally means a slightly better model
for real-world use, and we're no longer using the test set to make any
further decisions, so this doesn't reintroduce leakage. The two saved
`.pkl` files are therefore trained on all 3000 (classification) /
1487 (regression) rows, but their PERFORMANCE CLAIMS come from the earlier
held-out test evaluation, not from evaluating on data they were trained on.

WHAT GETS SAVED:
Complete sklearn Pipelines (preprocessing + model), not bare estimators.
This means the saved .pkl file can take raw, unscaled, unencoded input
(e.g. from the Streamlit form in Phase 11) and handle the full
transform-then-predict flow internally - the app never needs to duplicate
preprocessing logic.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
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
CATEGORICAL = ["branch"]


def build_pipeline(model):
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), RAW_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def finalize_classification(df):
    X = df[RAW_NUMERIC + CATEGORICAL]
    y = df["placement_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 1. Honest held-out evaluation
    pipe = build_pipeline(LogisticRegression(C=0.01, max_iter=2000, random_state=RANDOM_STATE))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, list(pipe.classes_).index("Placed")]
    final_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="Placed"),
        "recall": recall_score(y_test, y_pred, pos_label="Placed"),
        "f1": f1_score(y_test, y_pred, pos_label="Placed"),
        "roc_auc": roc_auc_score((y_test == "Placed").astype(int), y_proba),
    }

    # 2. Retrain on full data for deployment
    final_pipe = build_pipeline(LogisticRegression(C=0.01, max_iter=2000, random_state=RANDOM_STATE))
    final_pipe.fit(X, y)

    return final_pipe, final_metrics


def finalize_regression(df):
    placed = df[df["placement_status"] == "Placed"].copy()
    X = placed[RAW_NUMERIC + CATEGORICAL]
    y = placed["package_lpa"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 1. Honest held-out evaluation
    pipe = build_pipeline(Ridge(alpha=10, random_state=RANDOM_STATE))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    final_metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_test, y_pred),
    }

    # 2. Retrain on full (placed-only) data for deployment
    final_pipe = build_pipeline(Ridge(alpha=10, random_state=RANDOM_STATE))
    final_pipe.fit(X, y)

    return final_pipe, final_metrics


if __name__ == "__main__":
    df = pd.read_csv("data/placement_data_featured.csv")

    clf_pipe, clf_metrics = finalize_classification(df)
    reg_pipe, reg_metrics = finalize_regression(df)

    print("=== FINAL Placement Model (Logistic Regression, C=0.01) ===")
    print("Held-out test metrics (honest, from unseen 20% split):")
    for k, v in clf_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== FINAL Salary Model (Ridge Regression, alpha=10) ===")
    print("Held-out test metrics (honest, from unseen 20% split):")
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.4f}")

    joblib.dump(clf_pipe, "models/placement_model.pkl")
    joblib.dump(reg_pipe, "models/salary_model.pkl")
    print("\nSaved models/placement_model.pkl and models/salary_model.pkl")
    print("(both retrained on the FULL dataset - metrics above are from the earlier held-out split, not from these refit models)")
