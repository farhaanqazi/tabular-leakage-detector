# Data Leakage Detector: Tabular Feature Audit Tool

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-green)

This repository documents a **forensic data-leakage audit** of a Master's Project on Parkinson's Disease classification, and the reusable **Data Leakage Detector** it produced as a durable artifact.

The project began as a faithful reproduction of the original thesis. Reproducing its "too good to be true" 1.0 AUC results exposed a data leak — `Duration = 0.0` for every healthy control — that let models score perfectly by thresholding a single column. Rather than quietly patch that one dataset, the audit (a) honestly re-derives the true biomechanical performance, and (b) generalizes the pathology into a standalone, scoped detector that surfaces suspicious label-predictiveness in any tabular dataset prior to training.

The contribution is the audit discipline; the detector is the tool that fell out of it — deliberately small, scoped, and honest about its limits.

## 🎯 Scope & Capabilities

The detector scores features by evaluating univariate ROC-AUC, KS-statistic, and within-class variance. 
It frames leakage detection as **decision-support**, producing a ranked list of suspicious features for human review.

**What it reliably catches:**
- **Within-class degeneracy:** Features that are constant or near-constant within a specific class (e.g. the original Parkinson's `Duration` leak). This is caught via a strict hard-flag.
- **Extreme univariate correlation:** Surfaces exceptionally strong predictors (high AUC/KS) for domain review.

**What it does NOT catch (Out of Scope):**
- Temporal leakage (e.g. future data leaking into the past)
- Train/test set contamination
- Multivariate leakage (where no single feature looks individually suspicious)

## 📊 Dataset & Validation Suite

The detector is rigorously validated across multiple datasets to evaluate both precision and recall, ensuring it avoids false-positives on genuinely predictive features:

1. **Parkinson's Disease (Regression):** Re-catches the original `Duration` leak unprompted.
2. **Breast Cancer Wisconsin (Hard Negative Control):** Evaluates precision against legitimately strong clinical markers (AUC > 0.95), verifying they are not incorrectly flagged as degenerate.
3. **Wine Quality (Generalization):** Tests detection on binarized generic datasets with subtle injected proxy leaks.

See [DATA.md](docs/DATA.md) and [DATASHEET.md](docs/DATASHEET.md) for details on the primary PD dataset used for the core regression test.

## 📁 Repository Structure

```text
parkinsons-ml-repro/
├── README.md                  # overview + methodology + validation results
├── LICENSE                    # MIT
├── requirements.txt           # dependencies
├── Dockerfile                 # container setup
├── docs/
│   ├── DATA.md                # dataset source, version, access procedure
│   ├── DATASHEET.md           # Gebru et al. datasheet for the dataset
│   ├── MODEL_CARD.md          # Mitchell et al. model card
│   ├── TRIPOD-AI_checklist.md # Layer B clinical reporting
│   └── REPRODUCIBILITY.md     # Layer A reproducibility 
├── data/
│   ├── raw/                   # gitignored, contains dataset.csv
│   └── processed/             # cleaned features and labels
├── splits/                    # saved train/val/test indices + the seed
├── src/
│   ├── leakage_detector.py    # CORE: Standalone leakage detector class
│   ├── validate_detector.py   # CORE: Cross-dataset validation suite
│   ├── utils.py               # set_seed() for numpy / framework / split
│   ├── data_prep.py           # data exclusions and cleaning
│   ├── features.py            # standardization and explicit dataset splits
│   ├── train.py               # grid search and model saving
│   └── evaluate.py            # fairness, CIs, and metrics evaluation
├── models/                    # saved weights (.pkl)
├── results/                   # metrics.json, feature importances
└── notebooks/                 # original thesis notebooks
```

## 🚀 Exact Reproduction Commands

We provide a testing suite to validate the leakage detector across multiple datasets and evaluate its Precision-Recall curve.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the cross-dataset validation suite
python -m src.validate_detector
```

*(Optional) The original ML reproduction pipeline can still be run via `python src/train.py`.*

## 📈 1. Validation Results: Catching Data Leakage

### The Regression Case: Parkinson's Dataset
Initially, a Decision Tree achieved 1.0 accuracy on the PD dataset using a single split: `Duration > 0.0`. It was discovered that all Healthy Controls had a test duration of exactly `0.0`. The `DataLeakageDetector` successfully re-catches this issue automatically by flagging `Duration` with its **within-class degeneracy flag**, identifying the zero-variance pathology without human intervention.

### Precision Tests: Breast Cancer & Legitimate Predictiveness
A significant challenge for leakage detection is avoiding false-positives on highly predictive, legitimate features. When evaluated against Breast Cancer Wisconsin (where legitimate features like `worst perimeter` achieve `>0.97` AUC), the detector **correctly passes** the data without triggering degeneracy flags.

### Recall Tests: Subtly Injected Proxy Leaks
When injecting synthetic leaks (e.g. noisy proxies of the target) into clean datasets, the tool successfully ranks them near the top. However, because highly correlated proxy leaks are mathematically indistinguishable from legitimate high-AUC clinical markers, the PR curve behaves as expected:
- The detector successfully isolates degeneracy leaks (perfect precision).
- For pure correlation proxies, it highlights them as "suspicious" for human review. It does not attempt to falsely automate the separation between a strong true predictor and a correlated leak.

## 🛠️ 3. The Correction & Honest Evaluation

To properly reproduce the clinical value of the spatial-temporal features (Velocity, Acceleration, AreaError, etc.), the leaked `Duration` column was explicitly dropped from the feature set. The pipeline was re-run, yielding the following true performance metrics:

| Model | Accuracy | Accuracy (95% CI) | Precision | Sensitivity | AUC (95% CI) |
|-------|----------|-------------------|-----------|-------------|--------------|
| **Random Forest** | **0.831** | [0.738, 0.908] | 0.840 | 0.933 | **[0.646, 0.916]** |
| **XGBoost**       | 0.754 | [0.646, 0.846] | 0.774 | 0.911 | [0.618, 0.893] |
| **Decision Tree** | 0.754 | [0.646, 0.862] | 0.854 | 0.778 | [0.625, 0.872] |
| **Logistic Reg.** | 0.692 | [0.569, 0.800] | 0.727 | 0.889 | [0.675, 0.903] |
| **SVC**           | 0.677 | [0.569, 0.785] | 0.731 | 0.844 | [0.515, 0.784] |
| **KNN**           | 0.662 | [0.538, 0.769] | 0.735 | 0.800 | [0.438, 0.736] |
| **MLP**           | 0.646 | [0.538, 0.754] | 0.739 | 0.756 | [0.580, 0.838] |

### 📊 Final Synthesis: Before & After Data Leak Correction

This table directly compares the artificial metrics generated by the `Duration` leak against the honest biomechanical baseline:

| Model | Accuracy (Leaked) | Accuracy (Honest) | Accuracy Diff | AUC (Leaked) | AUC (Honest) |
|-------|-------------------|-------------------|---------------|--------------|--------------|
| Random Forest | 1.000 | 0.831 | -0.169 | 1.000 | 0.788 |
| XGBoost | 1.000 | 0.754 | -0.246 | 1.000 | 0.762 |
| Decision Tree | 1.000 | 0.754 | -0.246 | 1.000 | 0.747 |
| Logistic Reg. | 0.984 | 0.692 | -0.292 | 1.000 | 0.798 |
| SVC | 1.000 | 0.677 | -0.323 | 0.999 | 0.658 |
| KNN | 0.800 | 0.662 | -0.138 | 0.900 | 0.588 |
| MLP | 1.000 | 0.646 | -0.354 | 0.999 | 0.716 |

This realistic baseline (~75% - 83%) validates the original 2023 thesis conclusions that differentiating healthy controls is challenging using purely biomechanical features, and demonstrates the necessity of rigorous Data QA in medical MLOps.

### Subgroup Fairness

Fairness was audited using the proxy feature `Dominant`. On the honest evaluation, Random Forest accuracy showed a noticeable performance gap across subgroups (0.889 vs 0.759). This indicates a potential statistical bias along handedness/laterality that requires further clinical investigation before deployment.
