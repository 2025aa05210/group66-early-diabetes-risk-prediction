# Early Diabetes Risk Prediction System

## Software Engineering for Machine Learning (SEML) – Assignment 1

### Group 66

### Project Overview

The Early Diabetes Risk Prediction System is a Machine Learning-based healthcare application designed to predict the likelihood of diabetes using patient health information. The system assists healthcare professionals in identifying individuals at risk of diabetes and supports preventive healthcare decision-making.

The solution uses the Pima Indians Diabetes Dataset and applies multiple machine learning algorithms to classify patients as diabetic or non-diabetic based on diagnostic measurements.

---

## Problem Statement

Diabetes is one of the most common chronic diseases worldwide and often remains undiagnosed until severe symptoms or complications arise. Healthcare providers collect various patient health indicators, but accurately identifying high-risk individuals can be challenging.

This project aims to develop a Machine Learning-based Early Diabetes Risk Prediction System capable of analyzing patient health parameters and predicting diabetes risk to support healthcare professionals in making informed decisions.

---

## Dataset

**Dataset:** Pima Indians Diabetes Dataset

### Features

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

### Target Variable

* Outcome

  * 0 = Non-Diabetic
  * 1 = Diabetic

---

## Technologies Used

| Component               | Technology                      |
| ----------------------- | ------------------------------- |
| Programming Language    | Python 3.x                      |
| Data Processing         | Pandas, NumPy                   |
| Machine Learning        | Scikit-Learn                    |
| Visualization           | Matplotlib, Seaborn             |
| Web Application         | Streamlit                       |
| Development Environment | Jupyter Notebook / Google Colab |
| Version Control         | Git & GitHub                    |

---

## Project Structure

```text
Early-Diabetes-Risk-Prediction/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── Diabetes_Prediction.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── app/
│   └── app.py
│
├── docs/
│   ├── Business_View.docx
│   ├── Analytics_Design_View.docx
│   ├── Data_Preparation_View.docx
│   └── System_Architecture.docx
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Data Preparation

The dataset was prepared using the following steps:

* Data quality assessment
* Missing value handling
* Median imputation
* Data validation
* Feature scaling
* Train-test split

---

## Machine Learning Models Evaluated

The following machine learning algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

---

## Model Performance

| Model                        | Accuracy (%) |
| ---------------------------- | -----------: |
| Logistic Regression          |        75.32 |
| Support Vector Machine (SVM) |        74.68 |
| Random Forest                |        73.38 |
| Decision Tree                |        72.73 |
| K-Nearest Neighbors (KNN)    |        72.08 |

### Selected Model

**Logistic Regression**

Logistic Regression achieved the highest accuracy of 75.32% and was selected as the final model for deployment.

---

## System Architecture

The system follows a layered architecture consisting of:

1. User Interface Layer
2. Application Layer
3. Machine Learning Layer
4. Data Layer

The architecture supports maintainability, reliability, and scalability.

---

## Streamlit Application

The application provides:

* Patient data entry
* Input validation
* Diabetes prediction
* Risk classification
* Prediction result display

### Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Streamlit:

```bash
streamlit run app.py
```

---

## Future Enhancements

* Hyperparameter tuning
* Additional machine learning models
* Automated model retraining
* Cloud deployment
* Integration with healthcare systems
* Real-time prediction services

---

## Team Contributions

| Member    | Contribution                                                                                |
| --------- | ------------------------------------------------------------------------------------------- |
| Ashwin    | Domain Formulation, Requirements, Architecture                                              |
| Devi      | Business View, Analytics Design View, Data Preparation View                                 |
| Nandini   | Architecture Patterns, Dataset Preparation, Feature Engineering                             |
| Ajay Nath | Model Development, Model Evaluation, GitHub Repository Administration, Streamlit Deployment |

---

##

This project was developed for academic purposes as part of the Software Engineering for Machine Learning (SEML) course.
