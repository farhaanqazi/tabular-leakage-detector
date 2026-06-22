# Machine Learning Reproducibility Checklist (v2.0)

## Project Origin
This repository began as a faithful reproduction of a Master's thesis on Parkinson's disease classification. Reproducing the originally reported near-perfect performance exposed a data leakage artifact — the `Duration = 0.0` degeneracy for Healthy Controls (see [`DATA.md`](DATA.md)). That forensic discovery motivated the reusable `DataLeakageDetector` (`src/leakage_detector.py`), which is now run as a pre-modeling QA step and regression test. Reproducibility here therefore covers both the corrected ("honest") model results and the detector's cross-dataset validation.

## Models / Algorithms
- [x] Plain description of each model and its mathematical setup: Detailed in `MODEL_CARD.md` and training scripts.
- [x] Assumptions stated explicitly: Noted in `MODEL_CARD.md`.
- [x] Complexity noted where relevant: O(n) notation and empirical runtimes noted in results.

## Data
- [x] Dataset statistics: Reported in `DATASHEET.md`.
- [x] Exact train / validation / test splits: Indices saved in `splits/`.
- [x] Every exclusion and preprocessing step documented: See `src/data_prep.py`.
- [x] Link to dataset / access procedure: Detailed in `DATA.md`.
- [x] Collection / labeling / QC: N/A (secondary dataset).
- [x] Data leakage audit: `DataLeakageDetector` flags the `Duration` within-class degeneracy; the feature is dropped in `src/features.py` before training. Validation suite: `python -m src.validate_detector`.

## Code
- [x] Dependencies pinned: `requirements.txt` / `Dockerfile` provided.
- [x] Training code included: `src/train.py`.
- [x] Evaluation code included: `src/evaluate.py`.
- [x] Trained models included: Saved in `models/`.
- [x] README with exact reproduce commands: Provided in root `README.md`.

## Reported Results
- [x] Hyperparameter ranges / selection method: Detailed in `src/train.py`.
- [x] Exact number of training and evaluation runs: Logged in results.
- [x] Each metric precisely defined: Detailed in `MODEL_CARD.md`.
- [x] Results with central tendency and variation: Reported with 95% CIs.
- [x] Runtime / compute cost: Included in results.
- [x] Compute infrastructure described: Standard Python 3.9 Docker container.
