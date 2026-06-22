# TRIPOD+AI Clinical Reporting Checklist

## Title & Abstract
- [x] Identify as diagnostic prediction model.
- [x] Abstract structured with objective, data, model type, validation, performance.

## Introduction
- [x] Clinical background and rationale (Parkinson's diagnosis via movement/motor tests).
- [x] Specific objectives: Retrospective evaluation of classification models.

## Methods
- [x] Data source and study setting: Described in `DATASHEET.md`.
- [x] Participants: Eligibility and selection detailed.
- [x] Outcome (Parkinson's label): `PC` column (1 = Parkinson's, 0 = Healthy Control).
- [x] Predictors / features: Biomechanical properties (AreaError, TimeTriangles, ZeroVel, ZeroAcc, etc.).
- [x] Sample size and missing-data handling: N=325, no missing values assumed (or handled via simple imputation if present).
- [x] Feature processing & QA: Applied `DataLeakageDetector` (univariate AUC, KS stat, degeneracy check) prior to modeling to isolate and remove leaked predictors (e.g., `Duration`). Features were subsequently standardized.
- [x] Model type & development procedure: Standard ML classifiers; exact setup in code.
- [x] Validation approach: 80/20 random split (saved securely) + 5-fold cross-validation on train.
- [x] Performance measures: AUC (discrimination), calibration curves, and precision/recall (clinical utility).

## Fairness & Open Science (TRIPOD+AI Additions)
- [x] Fairness: Subgroup analysis on `Side` or `Dominant` attributes (surrogates for handedness/laterality).
- [x] Open science: Code, splits, and models available via GitHub.
- [x] ML-specific reporting: Feature importance via Random Forest Gini impurity.

## Results
- [x] Participant flow and characteristics: Summarized in `results/`.
- [x] Data Leakage Audit: Detected zero-variance degeneracy in control `Duration`; feature excised before model evaluation.
- [x] Full model specification: Available via `models/` directory and exact `scikit-learn` version.
- [x] Performance with uncertainty: 95% Confidence Intervals reported.
- [x] Subgroup / fairness results: Reported in `evaluate.py` outputs.

## Discussion
- [x] Limitations & PROBAST self-assessment: Retrospective bias, single-center limitations.
- [x] Usability and clinical implications: Motor-tests offer a non-invasive screening tool.
