"""
data_preprocessing.py
----------------------
Data CLEANING utilities (Phase 3) + reusable preprocessing pipeline builders
(used later in Phase 6/7 for model training).

Cleaning decisions and WHY:

1. Duplicate rows -> dropped entirely. They add no new information and can
   bias the model toward those repeated examples (and can leak between
   train/test splits if not removed first).

2. `twelfth_percentage`, `communication_score` -> these have a small amount
   (~1-2%) of genuinely missing values (simulating real-world "student didn't
   report this"). We fill these with the MEDIAN rather than the mean, because
   median is robust to skew/outliers - a few extreme values won't drag the
   fill value away from the "typical" student the way a mean could.
   NOTE: In the actual model training pipelines (Phase 6/7) this imputation
   will be done INSIDE a scikit-learn Pipeline fitted only on the training
   split - never on the full dataset before splitting - to avoid data
   leakage. Here in Phase 3 we do a "preview" clean on the full dataset only
   to understand the data; the real, leakage-safe imputation happens later.

3. `package_lpa` -> NOT imputed. It is missing by definition for every
   "Not Placed" student, since a student who wasn't placed has no salary.
   This is what's sometimes called MNAR (Missing Not At Random) - the
   missingness itself carries information (it's a function of the placement
   outcome). The correct handling is to FILTER to placed students only when
   building the salary regression dataset (Phase 7), never to impute a
   plausible-looking salary for someone who wasn't placed.

4. Dtypes -> verified as-is; `branch` and `placement_status` remain as
   strings/objects here and are encoded later inside the preprocessing
   Pipeline (OneHotEncoder / label encoding), not manually here, again to
   keep encoding logic inside the leakage-safe Pipeline.
"""

import pandas as pd


def load_raw_data(path: str = "data/placement_data.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs the cleaning steps that are safe to do on the full dataset
    (i.e. steps that don't involve fitting anything that could leak
    train/test information): dropping duplicates, imputing the two
    genuinely-missing columns with median, and validating dtypes.

    `package_lpa` is intentionally left untouched here - see module
    docstring point 3.
    """
    df = df.copy()

    # --- 1. Drop duplicate rows ---
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Dropped {before - after} duplicate rows ({before} -> {after})")

    # --- 2. Impute genuinely-missing columns with median ---
    for col in ["twelfth_percentage", "communication_score"]:
        n_missing = df[col].isnull().sum()
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {n_missing} missing values in '{col}' with median={median_val:.2f}")

    # --- 3. package_lpa: leave as-is, just report ---
    n_null_package = df["package_lpa"].isnull().sum()
    n_not_placed = (df["placement_status"] == "Not Placed").sum()
    assert n_null_package == n_not_placed, (
        "Mismatch: package_lpa nulls should exactly equal Not Placed count. "
        "If this fails, something other than 'not placed' is causing missing salaries."
    )
    print(
        f"'package_lpa' has {n_null_package} missing values, all corresponding "
        f"to 'Not Placed' students ({n_not_placed}) - left as-is by design."
    )

    # --- 4. Validate dtypes ---
    expected_numeric = [
        "cgpa", "iq_score", "tenth_percentage", "twelfth_percentage",
        "graduation_percentage", "internships", "projects",
        "technical_skills_score", "communication_score", "backlogs",
        "certifications", "work_experience_months", "package_lpa",
    ]
    for col in expected_numeric:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric but isn't"

    for col in ["branch", "placement_status"]:
        assert not pd.api.types.is_numeric_dtype(df[col]), f"{col} should be categorical/text but isn't"

    print("Dtype validation passed.")

    return df


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    cleaned.to_csv("data/placement_data_cleaned.csv", index=False)
    print(f"\nSaved cleaned data: data/placement_data_cleaned.csv, shape={cleaned.shape}")
