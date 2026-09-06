"""
train_classification.py
------------------------
Trains and compares classification models to predict `placement_status`.

KEY DESIGN DECISIONS (explained for interview prep too):

1. Train-test split happens FIRST, before any preprocessing is fit.
   All scaling/encoding/imputing is done inside an sklearn Pipeline that is
   .fit() only on the training split. This is the single most important
   anti-leakage rule in this project: if you fit a StandardScaler or
   OneHotEncoder on the full dataset before splitting, information about
   the test set's distribution leaks into training, and your test metrics
   become optimistic/unreliable.

2. `package_lpa` is EXCLUDED from the feature set for classification.
   Salary is only known/decided after a student is placed, so including it
   would leak the answer (a trivial model could just check "is package_lpa
   not null").

3. We compare WITH and WITHOUT the Phase 5 engineered features, since
   Phase 5's correlation check showed they don't obviously dominate the raw
   features - this is decided by evidence (the comparison table), not
   assumption.

4. Numeric features are scaled (StandardScaler) - needed for
   Logistic Regression, KNN, and SVM, harmless for tree-based models.
   `branch` is one-hot encoded via ColumnTransformer.

5. We report Accuracy, Precision, Recall, F1, and ROC-AUC - not just
   accuracy - because in a placement context, both false positives
   (telling a student they'll be placed when they won't) and false
   negatives (telling a placement-ready student they won't be placed) have
   real costs, so a single metric hides real tradeoffs.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

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
TARGET = "placement_status"


def build_pipeline(model, numeric_features):
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label="Placed"),
        "Recall": recall_score(y_true, y_pred, pos_label="Placed"),
        "F1 Score": f1_score(y_true, y_pred, pos_label="Placed"),
        "ROC-AUC": roc_auc_score((y_true == "Placed").astype(int), y_proba),
    }


def run_comparison(df: pd.DataFrame, numeric_features: list, label: str) -> pd.DataFrame:
    X = df[numeric_features + CATEGORICAL]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    fitted_pipelines = {}
    for name, model in get_models().items():
        pipe = build_pipeline(model, numeric_features)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, list(pipe.classes_).index("Placed")]
        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["Model"] = name
        results.append(metrics)
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(results).set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ].round(4)
    print(f"\n=== Classification comparison ({label}) ===")
    print(results_df.sort_values("F1 Score", ascending=False))

    return results_df, fitted_pipelines, (X_test, y_test)


if __name__ == "__main__":
    df = pd.read_csv("data/placement_data_featured.csv")

    results_raw, pipelines_raw, test_raw = run_comparison(df, RAW_NUMERIC, "raw features only")
    results_eng, pipelines_eng, test_eng = run_comparison(
        df, RAW_NUMERIC + ENGINEERED_NUMERIC, "raw + engineered features"
    )

    results_raw.to_csv("reports/classification_results_raw.csv")
    results_eng.to_csv("reports/classification_results_engineered.csv")

    print("\nSaved comparison tables to reports/")
