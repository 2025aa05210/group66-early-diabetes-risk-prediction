# Assignment II Architecture and Requirements Traceability

## Execution Flow

1. `src/data/loader.py` loads the diabetes dataset from the configured path.
2. `src/data/validator.py` validates schema and numeric types and measures
   missing values and medically invalid diagnostic zeros.
3. `src/features/preprocessing.py` loads the fitted scaler and transforms
   inference inputs.
4. `src/train.py` and `src/models/retrain_model.py` provide reproducible model
   training and quality metrics.
5. `src/models/predict.py` loads the serialized Logistic Regression model and
   returns a class and probability.
6. `api/schemas.py` validates API input and output contracts.
7. `api/service.py` orchestrates preprocessing and inference.
8. `api/main.py` exposes root, health, and prediction endpoints and converts
   unexpected failures into safe HTTP 500 responses.
9. `tests/` independently verifies data, preprocessing, training, inference,
   API, and integration behaviour.

## Requirements Traceability

| Assignment requirement | Implementation | Verification/evidence |
|---|---|---|
| Modular OOP/functional design | `src/`, `api/` | Repository structure and integration test |
| Research vs production comparison | Notebook and modular package | `reports/research_vs_production.md` |
| Logging and error handling in 3+ critical areas | Loader, validator, training, prediction, API | Pipeline logs and negative tests |
| Formatting and linting | Black, isort, flake8 | `lint_before.txt`, `lint_after.txt`, Black/isort reports |
| REST API | FastAPI root, health, and predict endpoints | API tests and Swagger evidence |
| At least two test types | Unit, data-validation, API, integration | Final pytest report |
| Model-training test | Small synthetic training dataset | `tests/test_training.py` |
| Model-inference test | Class, type, probability, failure, directional checks | `tests/test_inference.py` |
| Two model-quality metrics | Accuracy and F1 | `reports/model_metrics.txt` |
| Two data-quality metrics | Schema validity, missing-value rate, invalid-zero rate | Data-quality report and validator tests |
| Production experimentation | Shadow deployment | Security/production section in final report |
| Security consideration | Strict input validation and safe error/data handling | API schemas, error test, final report |

