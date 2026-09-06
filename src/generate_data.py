"""
generate_data.py
-----------------
Generates a SYNTHETIC student placement & salary dataset.

IMPORTANT: This data is NOT real. No suitable public dataset exists that
combines all the features this project needs (CGPA, IQ, 10th/12th %,
graduation %, internships, projects, skills, communication, backlogs,
work experience, certifications, branch) together with BOTH a placement
label AND a salary figure. Real placement datasets on Kaggle either have
the behavioral features but no salary, or have salary but only 4-5 basic
academic features.

So we simulate a dataset with realistic STRUCTURE:
    - Feature distributions are based on plausible ranges seen in real
      student data (e.g. CGPA ~4-10 with mean ~7, IQ ~85-140).
    - Placement probability is a weighted, noisy function of CGPA,
      internships, projects, skills, communication, and backlogs -
      mirroring how these factors are known to correlate with placement
      in practice, but NOT a deterministic formula (real noise is added
      so the ML problem is non-trivial and there is irreducible error,
      just like real-world data).
    - Salary is only generated for placed students, as a noisy function
      of CGPA, skills, internships, and branch.

This keeps the project statistically meaningful and honest, rather than
pretending this is a real-world dataset.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 3000

rng = np.random.default_rng(RANDOM_SEED)


def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_dataset(n=N_SAMPLES):
    # ---- Independent / mostly-independent raw features ----
    cgpa = clip(rng.normal(7.0, 1.1, n), 4.0, 10.0)
    iq_score = clip(rng.normal(100, 15, n), 70, 160)
    tenth_percentage = clip(rng.normal(78, 10, n), 40, 100)
    twelfth_percentage = clip(rng.normal(75, 10, n), 40, 100)
    graduation_percentage = clip(rng.normal(72, 8, n), 40, 100)

    internships = clip(rng.poisson(1.2, n), 0, 5)
    projects = clip(rng.poisson(2.5, n), 0, 8)
    technical_skills_score = clip(rng.normal(6.0, 2.0, n), 0, 10)
    communication_score = clip(rng.normal(6.2, 1.8, n), 0, 10)
    backlogs = clip(rng.poisson(0.5, n), 0, 6)
    certifications = clip(rng.poisson(1.0, n), 0, 6)

    # work experience: most freshers have 0, some have a few months
    has_exp = rng.random(n) < 0.15
    work_experience_months = np.where(
        has_exp, clip(rng.exponential(6, n), 1, 36), 0
    ).round().astype(int)

    branches = ["CSE", "IT", "ECE", "EE", "ME", "CE"]
    # slight base employability differences by branch (reflects real hiring
    # demand skew across branches, kept modest on purpose)
    branch_weight = {"CSE": 0.35, "IT": 0.30, "ECE": 0.10,
                      "EE": 0.0, "ME": -0.10, "CE": -0.15}
    branch = rng.choice(branches, size=n, p=[0.30, 0.20, 0.20, 0.10, 0.12, 0.08])
    branch_effect = np.array([branch_weight[b] for b in branch])

    # ---- Placement probability: weighted, standardized, + noise ----
    def z(x):
        return (x - x.mean()) / x.std()

    logit = (
        1.1 * z(cgpa)
        + 0.55 * z(internships)
        + 0.45 * z(projects)
        + 0.5 * z(technical_skills_score)
        + 0.35 * z(communication_score)
        - 0.6 * z(backlogs)
        + 0.2 * z(certifications)
        + 0.15 * z(iq_score)
        + branch_effect
        + rng.normal(0, 0.9, n)  # irreducible noise -> keeps problem non-trivial
        - 0.2  # slight negative bias so placement isn't ~50/50 by default
    )
    placement_prob = 1 / (1 + np.exp(-logit))
    placement_status = np.where(rng.random(n) < placement_prob, "Placed", "Not Placed")

    # ---- Salary (LPA): only for placed students ----
    base_salary = (
        3.5
        + 0.9 * (cgpa - 7)
        + 0.35 * technical_skills_score
        + 0.25 * internships
        + 0.15 * certifications
        + 0.4 * branch_effect * 5  # branch has bigger effect on pay than on placement
        + rng.normal(0, 1.1, n)
    )
    package_lpa = clip(base_salary, 2.5, 45)
    package_lpa = np.where(placement_status == "Placed", package_lpa.round(2), np.nan)

    df = pd.DataFrame({
        "cgpa": cgpa.round(2),
        "iq_score": iq_score.round(0).astype(int),
        "tenth_percentage": tenth_percentage.round(2),
        "twelfth_percentage": twelfth_percentage.round(2),
        "graduation_percentage": graduation_percentage.round(2),
        "internships": internships.astype(int),
        "projects": projects.astype(int),
        "technical_skills_score": technical_skills_score.round(2),
        "communication_score": communication_score.round(2),
        "backlogs": backlogs.astype(int),
        "certifications": certifications.astype(int),
        "work_experience_months": work_experience_months,
        "branch": branch,
        "placement_status": placement_status,
        "package_lpa": package_lpa,
    })

    # Inject a small, realistic amount of missingness + a few duplicate rows
    # so the cleaning phase (Phase 3) has genuine work to do.
    missing_idx = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[missing_idx, "communication_score"] = np.nan
    missing_idx2 = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[missing_idx2, "twelfth_percentage"] = np.nan

    dup_rows = df.sample(n=15, random_state=RANDOM_SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv("data/placement_data.csv", index=False)
    print(f"Saved data/placement_data.csv with shape {dataset.shape}")
