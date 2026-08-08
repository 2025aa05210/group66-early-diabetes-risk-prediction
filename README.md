# Early Diabetes Risk Prediction System

## Software Engineering for Machine Learning (SEML)

### Assignment 1 & Assignment 2

**Group 66**

---

# Project Overview

The **Early Diabetes Risk Prediction System** is a Machine Learning-based healthcare application designed to predict the likelihood of diabetes using patient clinical information. The system assists healthcare professionals in identifying individuals at risk of diabetes and supports preventive healthcare decision-making.

The solution uses the **Pima Indians Diabetes Dataset** and evaluates multiple machine learning algorithms before selecting **Logistic Regression** as the final deployment model.

Assignment 2 extends the Assignment 1 implementation by transforming the research prototype into a **production-ready machine learning application** using modular architecture, centralized logging, REST APIs, automated testing, and software engineering best practices.

---

# Problem Statement

Diabetes is one of the most common chronic diseases worldwide and often remains undiagnosed until severe symptoms or complications arise. Healthcare providers collect various patient health indicators, but accurately identifying high-risk individuals can be challenging.

This project develops a Machine Learning-based Early Diabetes Risk Prediction System capable of analyzing patient health parameters and predicting diabetes risk to support healthcare professionals in making informed decisions.

---

# Dataset

**Dataset:** Pima Indians Diabetes Dataset

## Features

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Target Variable

- **Outcome**
  - **0** = Non-Diabetic
  - **1** = Diabetic

---

# Technologies Used

| Component | Technology |
|------------|------------|
| Programming Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Web Application | Streamlit |
| REST API | FastAPI |
| Testing | Pytest |
| API Validation | Pydantic |
| Logging | Python Logging |
| Code Formatting | Black |
| Linting | Flake8 |
| Import Formatting | isort |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git & GitHub |

---

# Assignment 2 Enhancements

The Assignment 1 project has been refactored into a production-ready machine learning system by applying software engineering best practices.

## Production Code Refactoring

- Modular project architecture
- Separation of concerns
- Configuration management
- Reusable ML components
- Production-ready package structure

## Logging

Centralized logging has been implemented using Python's `logging` module.

Logging is available for:

- Dataset loading
- Dataset validation
- Data preprocessing
- Model loading
- Model training
- Model prediction
- Model evaluation
- REST API requests

Log levels used:

- INFO
- WARNING
- ERROR

Application logs are stored in:

```text
logs/app.log
```

---

## Error Handling

Custom exception handling has been implemented throughout the application.

Examples include:

- Dataset loading errors
- Model loading errors
- Prediction failures
- Training failures
- Invalid API requests

---

# REST API

A production-ready REST API has been implemented using **FastAPI**.

### Available Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | API Information |
| `GET /health` | Health Check |
| `POST /predict` | Predict Diabetes Risk |

Swagger Documentation:

```
http://localhost:8000/docs
```

---

# Research Code vs Production Code

## Research Code

- Jupyter Notebook
- Model experimentation
- Feature engineering
- Model comparison
- Data analysis

## Production Code

- Modular Python packages
- Reusable functions
- Configuration management
- Logging
- Exception handling
- REST API
- Automated testing

---

# Project Structure

```text
group66-early-diabetes-risk-prediction/
│
├── app.py
├── Diabetes_Prediction.ipynb
├── requirements.txt
│
├── data/
│   └── diabetes.csv
│
├── model/
│   ├── diabetes_model.pkl
│   └── scaler.pkl
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── service.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── validator.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── exceptions.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data_validation.py
│   ├── test_inference.py
│   ├── test_integration.py
│   ├── test_preprocessing.py
│   └── test_training.py
│
├── logs/
│   └── app.log
│
└── reports/
```
│
├── logs/
│
└── reports/
```

---

# Data Preparation

The dataset preparation includes:

- Data quality assessment
- Missing value handling
- Median imputation
- Dataset validation
- Feature scaling
- Train-test split (`random_state=42`)

---

# Machine Learning Models Evaluated

The following models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

---

# Model Performance

| Model | Accuracy (%) |
|------|-------------:|
| Logistic Regression | **75.32** |
| Support Vector Machine | 74.68 |
| Random Forest | 73.38 |
| Decision Tree | 72.73 |
| K-Nearest Neighbors | 72.08 |

## Selected Model

**Logistic Regression**

The model is evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score

The trained model and scaler are serialized using **Joblib** and reused by both the Streamlit application and FastAPI inference service.

---

# Quality Assurance

The project includes automated testing using **Pytest**.

### Unit Tests

- Dataset loading
- Dataset validation
- Data preprocessing
- Prediction module

### Integration Tests

- End-to-end prediction pipeline
- FastAPI endpoint testing

### Machine Learning Tests

- Model training
- Model inference
- Prediction output validation

### Data Quality Tests

- Schema validation
- Missing value detection

---

# Running the Project

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Run Streamlit Application

```bash
streamlit run app.py
```

---

## 3. Run FastAPI Server

```bash
uvicorn api.main:app --reload
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

---

## 4. Verify Prediction Pipeline

```bash
python test_pipeline.py
```

---

## 5. Run Automated Tests

```bash
pytest
```

---

## 6. Code Quality Checks

Format code:

```bash
black .
```

Check linting:

```bash
flake8
```

Sort imports:

```bash
isort .
```

---

# Future Enhancements

- Automated model retraining
- Model versioning
- Docker containerization
- CI/CD pipeline
- Kubernetes deployment
- Cloud deployment
- API authentication
- Model monitoring and drift detection

---

# Team Contributions

| Member | Contribution |
|---------|--------------|
| **Ashwin** | Domain Formulation, Requirements Engineering, System Architecture |
| **Devi** | Business View, Analytics Design View, Data Preparation View, Testing Support |
| **Nandini** | Architecture Patterns, Feature Engineering, Quality Assurance Support |
| **Ajay Nath** | Machine Learning Model Development, Production Code Refactoring, FastAPI Development, Logging, Exception Handling, Model Evaluation, GitHub Repository Administration, Streamlit Deployment, System Integration |

---

# Academic Note

This project was developed as part of the **Software Engineering for Machine Learning (SEML)** course at **BITS Pilani (WILP)**.

Assignment 1 focused on the development of the machine learning solution, while Assignment 2 extended the project into a **production-ready, maintainable, testable, and deployable machine learning system** by incorporating software engineering best practices such as modular architecture, centralized logging, REST APIs, automated testing, and quality assurance.
