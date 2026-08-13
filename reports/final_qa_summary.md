# Final Quality-Assurance Summary

Final verification was executed on 14 August 2026 using Python 3.12.13 and
scikit-learn 1.9.0.

## Automated Tests

- 27 tests collected.
- 27 tests passed.
- 100% measured coverage for the exercised `src` and `api` modules.
- Test categories: API, data validation, preprocessing, training, inference,
  and integration.

## Code Quality

- Black check: passed; 19 files unchanged after formatting.
- isort check: passed; imports correctly sorted.
- flake8 after formatting: passed; no violations.
- The pre-formatting flake8 report is retained for before/after evidence.

## Executed ML and Data Results

- Model accuracy: 0.7143.
- Model F1 score: 0.5600.
- Schema validity: 100.00%.
- Missing-value rate: 0.00%.
- Medically invalid diagnostic zeros: 652 (16.98% of checked cells).
- End-to-end prediction: class 1 with probability 0.7314.

The third-party deprecation warnings shown by pytest originate from installed
library internals and do not represent failed application tests.

