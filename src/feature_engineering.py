"""
feature_engineering.py
-----------------------
Creates engineered features on top of the cleaned dataset.

Engineered features and WHY:

1. `academic_score` = weighted average of cgpa (scaled to 100), 10th %,
   12th %, and graduation %.
   WHY: These four raw academic numbers are individually noisy/partial
   signals of "how strong is this student academically". A single combined
   score can be an easier, more stable signal for models (especially linear
   ones) than four separate correlated-ish inputs, and it's also a very
   natural, explainable feature to describe in an interview.

2. `skill_score` = weighted combination of technical_skills_score (already
   0-10) and certifications (scaled).
   WHY: Combines "how good are they" (skills score) with "how much effort
   did they put into proving it" (certifications) into one skill signal.

3. `experience_score` = weighted combination of internships, projects, and
   work_experience_months (scaled).
   WHY: All three represent practical, hands-on exposure. Combining them
   avoids the model having to separately learn near-identical weights for
   three features that all point at the same underlying idea ("practical
   exposure").

4. `overall_performance_score` = weighted combination of academic_score,
   skill_score, experience_score, communication_score, minus a penalty for
   backlogs.
   WHY: A single top-level "how strong a candidate is this student overall"
   feature - useful as an easy-to-explain feature and a good sanity-check
   feature (does it correlate with placement/package the way we'd expect?).

IMPORTANT - what we deliberately do NOT do:
- We do NOT use `package_lpa` (salary) as an input to ANY feature used for
  placement CLASSIFICATION. Salary is only decided/known *after* placement,
  so using it as a classification feature would leak the answer (a model
  could just learn "package_lpa is not null -> Placed", which is
  meaningless - you can't know a soon-to-be-placed student's future salary
  before they're placed).
- We do NOT use `placement_status` as an input feature when predicting
  `package_lpa` in regression - it's used only to FILTER the rows (placed
  students only), never as a model input, since it's a direct restatement
  of "this row has a salary at all".
- These engineered features are all built purely from features that exist
  BEFORE placement is known, so they're safe to use for classification.

IMPORTANT - about multicollinearity:
Because these composite features are built FROM other existing features
(e.g. academic_score is built from cgpa, tenth_percentage, etc.), they will
naturally correlate with their own components. This is expected and is NOT
data leakage (leakage is about using information from the future / from the
target; this is just redundant information restated differently) - but it
DOES create multicollinearity, which matters mainly for linear models
(Linear/Ridge/Lasso Regression, Logistic Regression), less so for
tree-based models (Random Forest, Gradient Boosting), which are robust to
correlated features.

Our approach: keep both the raw features and the engineered features
available, and let Phase 6/7 model comparison + feature importance decide
empirically whether the engineered features add value on top of the raw
ones, rather than assuming it upfront. We DO check the correlation below so
this decision is evidence-based, not guesswork.
"""

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Academic score (scaled to 0-100) ---
    df["academic_score"] = (
        (df["cgpa"] / 10 * 100) * 0.40
        + df["tenth_percentage"] * 0.20
        + df["twelfth_percentage"] * 0.20
        + df["graduation_percentage"] * 0.20
    )

    # --- Skill score (scaled to 0-10) ---
    # certifications capped at 5 for scoring purposes (diminishing returns
    # beyond 5 certifications - a reasonable, explainable design choice)
    df["skill_score"] = (
        df["technical_skills_score"] * 0.7
        + df["certifications"].clip(upper=5) / 5 * 10 * 0.3
    )

    # --- Experience score (scaled to 0-10) ---
    # internships capped at 3, projects capped at 6, work_exp capped at 24
    # months for scoring - again diminishing returns beyond typical ranges
    df["experience_score"] = (
        (df["internships"].clip(upper=3) / 3 * 10) * 0.45
        + (df["projects"].clip(upper=6) / 6 * 10) * 0.35
        + (df["work_experience_months"].clip(upper=24) / 24 * 10) * 0.20
    )

    # --- Overall performance score (scaled to 0-10) ---
    df["overall_performance_score"] = (
        (df["academic_score"] / 100 * 10) * 0.35
        + df["skill_score"] * 0.30
        + df["experience_score"] * 0.25
        + df["communication_score"] * 0.10
        - df["backlogs"].clip(upper=4) * 0.25  # small penalty per backlog
    ).clip(lower=0)

    return df


def check_feature_correlation(df: pd.DataFrame):
    """Reports how the new engineered features correlate with each other,
    with their own raw components, and with the regression target - purely
    for evidence-based feature selection later, not a decision made here."""
    engineered = ["academic_score", "skill_score", "experience_score",
                  "overall_performance_score"]

    print("Correlation of engineered features with package_lpa:")
    print(df[engineered + ["package_lpa"]].corr()["package_lpa"].drop("package_lpa").round(3))

    print("\nCorrelation of academic_score with its own raw components "
          "(expected to be high - by construction, not leakage):")
    print(df[["academic_score", "cgpa", "tenth_percentage",
               "twelfth_percentage", "graduation_percentage"]].corr()["academic_score"].round(3))


if __name__ == "__main__":
    df = pd.read_csv("data/placement_data_cleaned.csv")
    df = add_engineered_features(df)
    check_feature_correlation(df)
    df.to_csv("data/placement_data_featured.csv", index=False)
    print(f"\nSaved data/placement_data_featured.csv, shape={df.shape}")
