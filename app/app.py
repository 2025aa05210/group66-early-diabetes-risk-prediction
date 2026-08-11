import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Early Diabetes Risk Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------------
# Load Model
# ----------------------------------

try:
    model = joblib.load("model/diabetes_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ----------------------------------
# Header
# ----------------------------------

st.title("🩺 Early Diabetes Risk Prediction System")

st.markdown("""
### Software Engineering for Machine Learning (SEML)
### Assignment II
**Group 66**

This application predicts the likelihood of diabetes using patient clinical and demographic information.

**Final Model Selected:** Logistic Regression

**Dataset:** Pima Indians Diabetes Dataset

### Assignment II Enhancements
- Modular Machine Learning Pipeline
- FastAPI REST API
- Centralized Logging
- Exception Handling
- Automated Testing
- Production-Ready Architecture
""")

# ----------------------------------
# Sidebar
# ----------------------------------

st.sidebar.header("Project Information")

st.sidebar.info("""
**SEML Assignment II**

**Early Diabetes Risk Prediction System**

**Group 66**

### Group Members
- Ajay Nath
- Nandini Agarwal
- Devi Kampalli
- Ashwin Megha

### Technology Stack
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- FastAPI
- Pydantic
- Pytest
""")

# ----------------------------------
# Model Summary
# ----------------------------------

with st.expander("Model Development Summary", expanded=False):

    st.markdown("""
Five machine learning algorithms were evaluated for early diabetes risk prediction:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Logistic Regression was selected as the final deployment model because it provides a good balance between predictive performance, interpretability, and computational efficiency. As part of Assignment II, the trained model has been integrated into a production-ready application with modular architecture, centralized logging, REST API support, and automated testing.
""")

    comparison_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Support Vector Machine (SVM)",
            "Random Forest",
            "Decision Tree",
            "K-Nearest Neighbors (KNN)"
        ],
        "Accuracy (%)": [
            75.32,
            74.68,
            73.38,
            72.73,
            72.08
        ]
    })

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

# ----------------------------------
# Input Section
# ----------------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=1000,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

# ----------------------------------
# Prediction
# ----------------------------------

if st.button("Predict Diabetes Risk"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ High Risk of Diabetes")

        st.metric(
            "Risk Probability",
            f"{probability:.2%}"
        )

        st.warning(
            "The patient may be at risk of diabetes. "
            "Further medical evaluation is recommended."
        )

    else:

        st.success("✅ Low Risk of Diabetes")

        st.metric(
            "Risk Probability",
            f"{probability:.2%}"
        )

        st.info(
            "The patient appears to have a lower risk of diabetes."
        )

    st.subheader("Input Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree Function",
            "Age"
        ],
        "Value": [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# ----------------------------------
# Footer
# ----------------------------------

st.markdown("---")

st.caption(
    """
Software Engineering for Machine Learning (SEML) – Assignment II | Group 66

This system is intended as a clinical decision-support tool only and should not replace professional medical judgement.
"""
)