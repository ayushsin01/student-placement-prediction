# Student Placement & Salary Prediction System

An end-to-end machine learning project that predicts (1) whether a student will be placed and (2) the expected salary package for placed students, built with a full ML pipeline from data generation through Streamlit deployment.

## Problem Statement

Colleges and students often want an early, data-driven estimate of placement likelihood and expected compensation based on academic performance, skills, and experience. This project builds two supervised ML models to address that: a classifier for placement outcome and a regressor for salary package.

## Project Objective

- Predict `placement_status` (Placed / Not Placed) — **classification**
- Predict `package_lpa` for placed students — **regression**
- Demonstrate a complete, leakage-safe ML workflow: EDA → feature engineering → model comparison → hyperparameter tuning → deployment

## Features

- Interactive Streamlit app with Home, Placement Prediction, and Salary Prediction pages
- 6 classification algorithms and 6 regression algorithms compared head-to-head
- Hyperparameter tuning via `GridSearchCV` with 5-fold cross-validation
- Full leakage-prevention: train-test split before any preprocessing fit, `package_lpa` excluded from classification features, `placement_status` used only to filter (never as a regression feature)
- Reusable, documented EDA and preprocessing modules

## Dataset

**This dataset is synthetic — not real student records.** No public dataset combines every feature this project needed (CGPA, IQ, academic percentages, internships, projects, skills, communication, backlogs, work experience, certifications, branch) together with both a placement outcome and a salary figure — real placement datasets on Kaggle either have the behavioral features with no salary, or salary with only 4–5 basic academic features.

The dataset was generated (`src/generate_data.py`) with realistic, documented statistical structure — not random noise:
- Feature ranges based on plausible real-world values (e.g. CGPA 4–10, IQ 70–160)
- Placement probability as a weighted, noisy logistic function of CGPA, internships, projects, skills, communication, and backlogs
- Salary generated only for placed students, as a noisy function of CGPA, skills, internships, and branch
- Realistic imperfections deliberately included: ~1–2% missing values in two columns, and duplicate rows (cleaned in Phase 3)

**Rows:** 3,015 generated (3,000 after removing 15 duplicate rows) | **Columns:** 15

| Column | Meaning |
|---|---|
| `cgpa` | Cumulative GPA (4.0–10.0 scale) |
| `iq_score` | Aptitude/IQ test score |
| `tenth_percentage` / `twelfth_percentage` / `graduation_percentage` | Academic percentages |
| `internships` | Number of internships completed |
| `projects` | Number of academic/personal projects |
| `technical_skills_score` | Self/test-assessed technical skill score (0–10) |
| `communication_score` | Communication skill score (0–10) |
| `backlogs` | Number of academic backlogs |
| `certifications` | Number of certifications earned |
| `work_experience_months` | Prior work experience in months |
| `branch` | Engineering branch (CSE, IT, ECE, EE, ME, CE) |
| `placement_status` | Target 1 — Placed / Not Placed |
| `package_lpa` | Target 2 — Package in LPA (only for placed students) |

## Technologies Used

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit · Joblib

## ML Algorithms

**Classification:** Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Gradient Boosting
**Regression:** Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting

## Data Preprocessing

- Duplicate rows dropped (15 removed)
- Genuinely-missing values (`twelfth_percentage`, `communication_score`, ~1–2%) imputed with median
- `package_lpa` missingness left untouched for "Not Placed" students — it's structural (MNAR), not random, and is handled by filtering, not imputation
- All scaling (`StandardScaler`) and encoding (`OneHotEncoder`) done inside `sklearn.Pipeline` + `ColumnTransformer`, fit only on the training split, to prevent leakage

## EDA

Full univariate, bivariate, and multivariate analysis in `notebooks/placement_analysis.ipynb`, including:
- Distribution plots for CGPA, package, placement status, internships, projects, backlogs
- Boxplots comparing key features across placement outcomes
- Correlation heatmap confirming no meaningful multicollinearity among predictors and no leakage signal

## Model Evaluation

**Classification metrics:** Accuracy, Precision, Recall, F1 Score, ROC-AUC (not accuracy alone, since both false positives and false negatives carry real costs)

**Regression metrics:** MAE, MSE, RMSE, R² (RMSE/MAE reported in LPA for direct interpretability)

## Results

### Classification comparison (raw features, held-out test set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.717 | 0.718 | 0.704 | 0.711 | **0.791** |
| SVM | 0.692 | 0.692 | 0.680 | 0.686 | 0.771 |
| Gradient Boosting | 0.690 | 0.691 | 0.677 | 0.684 | 0.757 |
| Random Forest | 0.680 | 0.677 | 0.677 | 0.677 | 0.752 |
| KNN | 0.650 | 0.638 | 0.677 | 0.657 | 0.686 |
| Decision Tree | 0.603 | 0.599 | 0.599 | 0.599 | 0.603 |

### Regression comparison (raw features, held-out test set)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | 0.870 | 1.080 | **0.592** |
| Ridge Regression | 0.870 | 1.081 | 0.592 |
| Gradient Boosting | 0.907 | 1.125 | 0.558 |
| Random Forest | 0.957 | 1.187 | 0.508 |
| Lasso Regression (default α) | 1.330 | 1.693 | -0.001 |
| Decision Tree | 1.367 | 1.716 | -0.028 |

### After hyperparameter tuning

Tuning did not meaningfully change the top classification models (already near their ceiling given the dataset's built-in noise), but had a dramatic effect on regression: **Lasso's R² improved from -0.001 to 0.590** simply by tuning `alpha` from its default (1.0) down to 0.01 — a concrete demonstration of why hyperparameters matter, not just algorithm choice.

### Final selected models

| Task | Model | Final held-out metric |
|---|---|---|
| Placement | Logistic Regression (C=0.01) | ROC-AUC = 0.789, Accuracy = 70.2% |
| Salary | Ridge Regression (alpha=10) | R² = 0.591, RMSE = ±1.08 LPA |

Logistic Regression was chosen over Gradient Boosting for placement (better ROC-AUC, more interpretable coefficients). Ridge was chosen over Lasso for salary (near-identical R², keeps all features with small shrinkage rather than zeroing any out).

## Project Architecture

```text
student-placement-prediction/
│
├── data/
│   ├── placement_data.csv              # raw synthetic data
│   ├── placement_data_cleaned.csv      # after Phase 3 cleaning
│   └── placement_data_featured.csv     # after Phase 5 feature engineering
│
├── notebooks/
│   └── placement_analysis.ipynb        # data understanding + EDA
│
├── src/
│   ├── generate_data.py                # synthetic dataset generator
│   ├── data_preprocessing.py           # cleaning
│   ├── eda.py                          # reusable EDA plotting functions
│   ├── feature_engineering.py          # composite feature creation
│   ├── train_classification.py         # classification model comparison
│   ├── train_regression.py             # regression model comparison
│   ├── hyperparameter_tuning.py        # GridSearchCV tuning
│   └── train_final_models.py           # final training + model saving
│
├── models/
│   ├── placement_model.pkl             # full sklearn Pipeline
│   └── salary_model.pkl                # full sklearn Pipeline
│
├── reports/                            # saved comparison tables & plots
│
├── app.py                              # Streamlit application
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

```bash
# 1. Clone the repo and install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate the dataset and models from scratch
python src/generate_data.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_final_models.py

# 3. Launch the app
streamlit run app.py
```

## Screenshots

*(Add screenshots of the Home, Placement Prediction, and Salary Prediction pages here once you've run the app.)*

## Future Improvements

- Replace the synthetic dataset with real institutional placement data if/when available
- Add SHAP-based explainability to the Streamlit app so predictions show *why*, not just *what*
- Add a batch-prediction mode (CSV upload) for processing multiple students at once
- Track model performance over time and retrain on a schedule if deployed with real, evolving data
- Add authentication and logging if used in a real institutional setting

## Author

*(Your name here)*
