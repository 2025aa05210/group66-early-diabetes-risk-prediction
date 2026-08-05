# Data Validation and Data-Quality Metrics

## 1. Objective

The production-style ML pipeline validates the Pima Indians Diabetes
dataset before preprocessing, model training or inference. The purpose
is to detect structural and content-related problems early and to
produce measurable data-quality evidence for SEML Assignment II.

The implementation is available in:

`src/data/validator.py`

## 2. Validation Rules

The validator performs the following checks:

1. The dataset must not be empty.
2. All nine required columns must be present.
3. Every required column must contain numeric data.
4. Missing values are counted and measured.
5. Physiologically implausible zero values are measured in selected
   diagnostic columns.

The required columns are:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

Zero is considered a valid value for `Pregnancies` and `Outcome`.
However, zero is treated as a data-quality problem in the following
diagnostic measurements:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

## 3. Error-Handling Policy

The validator raises `DataValidationError` when:

- The dataset is empty.
- One or more required columns are missing.
- A required column contains non-numeric data.

Missing values and medically invalid zeros are logged as warnings and
included in the returned quality measurements. This permits their
extent to be measured before an explicit downstream treatment or
imputation policy is applied.

## 4. Data-Quality Metrics

Three data-quality measurements are implemented.

### 4.1 Schema Validity

Schema validity measures the percentage of required columns present:

`Schema validity (%) = Present required columns / Total required columns × 100`

### 4.2 Missing-Value Rate

The missing-value rate measures the proportion of dataset cells that
contain null values:

`Missing-value rate (%) = Missing values / Total dataset cells × 100`

### 4.3 Invalid-Zero Rate

The invalid-zero rate measures zero values across the five diagnostic
columns where zero is physiologically implausible:

`Invalid-zero rate (%) = Invalid diagnostic zeros / Checked diagnostic cells × 100`

## 5. Results on the Production Dataset

| Metric | Result |
|---|---:|
| Row count | 768 |
| Column count | 9 |
| Schema validity | 100.00% |
| Missing-value count | 0 |
| Missing-value rate | 0.00% |
| Invalid-zero count | 652 |
| Invalid-zero rate | 16.98% |

### Invalid Zeros by Column

| Column | Invalid-zero count |
|---|---:|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |
| **Total** | **652** |

## 6. Verification Scenarios

The validator was independently exercised using the following
scenarios:

| Scenario | Expected behaviour | Result |
|---|---|---|
| Original production dataset | Return quality metrics | Passed |
| Required `Glucose` column removed | Raise `DataValidationError` | Passed |
| Non-numeric value in `Glucose` | Raise `DataValidationError` | Passed |
| Empty DataFrame | Raise `DataValidationError` | Passed |
| Two missing values inserted | Report count 2 and rate 0.03% | Passed |
| Existing end-to-end pipeline | Complete prediction successfully | Passed |

The end-to-end regression check continued to return:

- Predicted class: 1
- Predicted probability: 0.7881

This confirms that the data-validation enhancement did not break the
existing prediction pipeline.

## 7. Interpretation

The dataset satisfies the required schema and contains no null values
in its original form. However, 652 zero values were identified across
the five diagnostic measurements. These represent 16.98% of all
checked diagnostic cells.

The largest contribution comes from `Insulin` and `SkinThickness`.
These values should therefore be handled through an explicit and
documented preprocessing or imputation policy before future model
retraining.

## 8. Testing Handoff

The manual verification scenarios provide the basis for formal pytest
tests covering:

- Valid dataset behaviour
- Missing required columns
- Non-numeric values
- Empty datasets
- Missing-value metrics
- Invalid-zero metrics

The formal automated tests will be maintained separately under the
project's `tests/` directory.
