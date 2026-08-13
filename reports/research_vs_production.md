# Research Code Versus Production-Style Code

## Purpose

Assignment I used a Jupyter notebook to explore the Pima Indians Diabetes
dataset, compare five algorithms, and select Logistic Regression. Assignment II
retains that research baseline and separates the operational responsibilities
into reusable modules that can be validated and tested independently.

## Comparison

| Aspect | Research notebook | Production-style implementation |
|---|---|---|
| Primary objective | Exploration and model comparison | Repeatable training and inference |
| Execution | Sequential notebook cells | Importable Python modules |
| Data access | Inline CSV loading | `src/data/loader.py` |
| Validation | Exploratory checks | `src/data/validator.py` with explicit exceptions and metrics |
| Preprocessing | Notebook transformations | `src/features/preprocessing.py` |
| Training | Inline model cells | `src/train.py` and `src/models/retrain_model.py` |
| Inference | Notebook predictions | `src/models/predict.py` |
| User access | Streamlit demonstration | Streamlit plus FastAPI REST endpoints |
| Failure handling | Cell errors | Custom exceptions, safe API responses, and structured logging |
| Verification | Manual inspection | Pytest unit, API, integration, training, inference, and data tests |
| Code quality | Notebook-oriented formatting | Black, isort, and flake8 evidence |
| Reproducibility | Recorded experiment | Fixed seed and pinned dependencies |

## Interpretation

The notebook remains valuable because it records the reasoning, exploration,
and model-selection history. The modular package serves a different purpose:
it makes responsibilities explicit, enables isolated tests, exposes a stable
API contract, and provides operational evidence through logs and quality
reports. The notebook is therefore retained as research evidence rather than
being replaced or artificially reformatted.

