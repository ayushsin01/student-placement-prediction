"""
app.py
------
Streamlit application for the Student Placement & Salary Prediction System.

This file ONLY loads the already-trained pipelines (from Phase 10) and does
inference. It never retrains anything - training lives entirely in src/,
which is exactly what you'd say in an interview when asked how the app
relates to the training code (see Phase 1's project structure reasoning).
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Student Placement & Salary Prediction System",
    page_icon="🎓",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_resource
def load_models():
    placement_model = joblib.load(os.path.join(MODEL_DIR, "placement_model.pkl"))
    salary_model = joblib.load(os.path.join(MODEL_DIR, "salary_model.pkl"))
    return placement_model, salary_model


BRANCHES = ["CSE", "IT", "ECE", "EE", "ME", "CE"]


def get_user_input():
    """Renders the shared input form and returns a single-row DataFrame."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Academics**")
        cgpa = st.slider("CGPA", 4.0, 10.0, 7.0, 0.1)
        tenth_percentage = st.slider("10th Percentage", 40.0, 100.0, 75.0, 0.5)
        twelfth_percentage = st.slider("12th Percentage", 40.0, 100.0, 75.0, 0.5)
        graduation_percentage = st.slider("Graduation Percentage", 40.0, 100.0, 70.0, 0.5)

    with col2:
        st.markdown("**Skills & Aptitude**")
        iq_score = st.slider("IQ / Aptitude Score", 70, 160, 100)
        technical_skills_score = st.slider("Technical Skills Score (0-10)", 0.0, 10.0, 6.0, 0.1)
        communication_score = st.slider("Communication Score (0-10)", 0.0, 10.0, 6.0, 0.1)
        branch = st.selectbox("Branch", BRANCHES)

    with col3:
        st.markdown("**Experience**")
        internships = st.number_input("Internships", 0, 10, 1)
        projects = st.number_input("Projects", 0, 15, 2)
        backlogs = st.number_input("Backlogs", 0, 10, 0)
        certifications = st.number_input("Certifications", 0, 10, 1)
        work_experience_months = st.number_input("Work Experience (months)", 0, 60, 0)

    return pd.DataFrame([{
        "cgpa": cgpa,
        "iq_score": iq_score,
        "tenth_percentage": tenth_percentage,
        "twelfth_percentage": twelfth_percentage,
        "graduation_percentage": graduation_percentage,
        "internships": internships,
        "projects": projects,
        "technical_skills_score": technical_skills_score,
        "communication_score": communication_score,
        "backlogs": backlogs,
        "certifications": certifications,
        "work_experience_months": work_experience_months,
        "branch": branch,
    }])


def home_page():
    st.title("🎓 Student Placement & Salary Prediction System")
    st.markdown(
        """
        This system predicts two things for a student profile:

        1. **Placement Prediction** — will the student likely be placed? (classification)
        2. **Salary Prediction** — if placed, what package can they expect? (regression)

        Use the sidebar to navigate between pages.
        """
    )

    st.subheader("How the system works")
    st.markdown(
        """
        - A **Logistic Regression** model predicts placement likelihood from academic,
          skill, and experience features.
        - A **Ridge Regression** model predicts the expected package (in LPA), trained
          only on historically placed students.
        - Both models were selected after comparing 6 classification and 6 regression
          algorithms, then tuned with `GridSearchCV`.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Placement Model ROC-AUC", "0.789")
        st.metric("Placement Model Accuracy", "70.2%")
    with col2:
        st.metric("Salary Model R²", "0.591")
        st.metric("Salary Model RMSE", "±1.08 LPA")

    with st.expander("⚠️ About the dataset"):
        st.markdown(
            """
            This project uses a **synthetic dataset** (not real student records). No
            public dataset combines all the features this project needs (CGPA, IQ,
            academic percentages, internships, projects, skills, communication,
            backlogs, work experience, certifications, branch) together with both a
            placement outcome and a salary figure. The synthetic data was generated
            with realistic, documented statistical relationships — see
            `src/generate_data.py`.
            """
        )

    st.subheader("Technologies used")
    st.markdown(
        "Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit"
    )

    st.subheader("ML workflow")
    st.markdown(
        "Data Generation → Cleaning → EDA → Feature Engineering → Model Training "
        "→ Comparison → Hyperparameter Tuning → Final Model → Deployment"
    )


def placement_page(placement_model):
    st.title("📋 Placement Prediction")
    st.write("Enter a student's profile to predict placement likelihood.")

    input_df = get_user_input()

    if st.button("Predict Placement", type="primary"):
        prediction = placement_model.predict(input_df)[0]
        proba = placement_model.predict_proba(input_df)[0]
        placed_idx = list(placement_model.classes_).index("Placed")
        placed_prob = proba[placed_idx]

        st.divider()
        if prediction == "Placed":
            st.success(f"### Placement Prediction: PLACED ✅")
        else:
            st.error(f"### Placement Prediction: NOT PLACED ❌")

        st.metric("Confidence (probability of placement)", f"{placed_prob*100:.1f}%")
        st.progress(float(placed_prob))

        # Store in session state so the salary page can reuse this profile
        st.session_state["last_input"] = input_df
        st.session_state["last_prediction"] = prediction

        if prediction == "Placed":
            st.info("Head to the **Salary Prediction** page to estimate the expected package for this profile.")


def salary_page(salary_model):
    st.title("💰 Salary Prediction")
    st.write("Predicts the expected package for a student assumed/predicted to be placed.")

    prefill = st.session_state.get("last_input")
    if prefill is not None:
        st.caption("Pre-filled from your last Placement Prediction input. Adjust if needed.")

    input_df = get_user_input()

    if st.button("Predict Package", type="primary"):
        salary_pred = salary_model.predict(input_df)[0]
        salary_pred = max(salary_pred, 0)  # guard against nonsensical negative output

        st.divider()
        st.success(f"### Predicted Package: ₹{salary_pred:.2f} LPA")
        st.caption(
            f"This model has a typical error (RMSE) of about ±₹1.08 LPA on held-out "
            f"test data — treat this as an estimate, not an exact figure."
        )


def main():
    placement_model, salary_model = load_models()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Placement Prediction", "Salary Prediction"],
    )

    if page == "Home":
        home_page()
    elif page == "Placement Prediction":
        placement_page(placement_model)
    elif page == "Salary Prediction":
        salary_page(salary_model)


if __name__ == "__main__":
    main()
