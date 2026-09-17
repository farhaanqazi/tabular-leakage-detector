# Data Leakage Detector: Tabular Feature Audit Tool

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-green)

During my 2023 MSc thesis on Parkinson's disease classification, I manually identified and excluded a structurally leaked variable (`Duration=0.0` for all healthy controls) that was causing artificial perfect-separation and overfitting in early model iterations. By manually dropping this column, I was able to report honest, rigorous biomechanical performance metrics for the thesis.

However, recognising how easily such structural flaws can silently compromise medical datasets, **that manual discovery inspired this project**: a reusable, automated **Data Leakage Detector** for tabular datasets. 

Instead of relying on a data scientist's manual intuition to notice "overfitting" or suspiciously high scores, this tool systematically evaluates and flags suspicious label-predictiveness in any tabular dataset *prior* to training. To demonstrate the tool's efficiency, it was tested on multiple real-world datasets—including the original Parkinson's dataset, where it successfully and automatically flagged the exact same `Duration` leak that I had to find manually in 2023.

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

1. **Parkinson's Disease (Regression):** The original inspiration. The tool successfully re-catches the `Duration` leak automatically.
2. **Breast Cancer Wisconsin (Hard Negative Control):** Evaluates precision against legitimately strong clinical markers (AUC > 0.95), verifying they are not incorrectly flagged as degenerate.
3. **Other Real-World Leaks:** Tests detection on Bank Marketing, Heart Failure, and Cervical Cancer datasets, all of which contain organic, documented proxy leaks.

See [DATA.md](docs/DATA.md) and [DATASHEET.md](docs/DATASHEET.md) for details on the primary PD dataset used for the core regression test.

## 📁 Repository Structure

```text
tabular-leakage-detector/
├── README.md                  # overview + methodology + validation results
├── LICENSE                    # MIT
├── requirements.txt           # dependencies
├── Dockerfile                 # container setup
├── docs/
│   ├── DATA.md                # dataset source, version, access procedure
│   ├── DATASHEET.md           # Gebru et al. datasheet for the dataset
│   ├── MODEL_CARD.md          # Mitchell et al. model card
│   ├── TRIPOD-AI_checklist.md # clinical reporting (Parkinson's case study)
│   └── REPRODUCIBILITY.md     # reproducibility checklist
├── data/
│   ├── raw/                   # gitignored; primary Parkinson's dataset.csv
│   ├── external/             # gitignored CSVs + download guide (README.md)
│   └── processed/            # cleaned features and labels
├── splits/                    # saved train/val/test indices + the seed
├── src/
│   ├── leakage_detector.py    # CORE: standalone leakage detector class
│   ├── validate_detector.py   # CORE: cross-dataset leak-impact suite
│   ├── utils.py               # set_seed() for numpy / framework / split
│   ├── data_prep.py           # data exclusions and cleaning
│   ├── features.py            # standardization and explicit dataset splits
│   ├── train.py               # grid search and model saving
│   └── evaluate.py            # fairness, CIs, and metrics evaluation
├── models/                    # saved weights (.pkl)
├── results/                   # metrics.json, leak_impact.json, charts
└── notebooks/
    ├── hotel_booking_leakage.ipynb   # rigorous per-dataset case study (gold template)
    ├── Masters_Project_F21MP.ipynb   # original thesis: main analysis + EDA
    ├── Outcome_Visualisation.ipynb   # original thesis: result visualisations
    ├── Comparison.csv                 # original thesis: model comparison
    └── *.ipynb                        # original thesis per-model notebooks:
                                       #   Decision_Trees, Random_Forest, Random_Forest_+_PSO,
                                       #   SVM, KNN, Logistic_Regression, MLP
```

## 🚀 Exact Reproduction Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the external validation datasets into data/external/
#    (filenames, sources, and known leaks listed in data/external/README.md)

# 3. Run the cross-dataset leak-impact suite (uniform baseline models)
python -m src.validate_detector
```

Two layers of evidence:
- **`src/validate_detector.py`** — fast, uniform-baseline sweep across all datasets (the summary tables below).
- **`notebooks/`** — full, rigorous per-dataset case studies (preprocessing, feature engineering, 7 tuned models, CIs) that *independently rediscover* each leak, then cross-check the detector. See [notebooks/hotel_booking_leakage.ipynb](notebooks/hotel_booking_leakage.ipynb) for the worked template.

*(Optional) The original Parkinson's ML reproduction pipeline can still be run via `python src/train.py`.*

## 📈 Validation Results: Catching Real Leaks Across 5 Datasets

The detector is validated on **five real-world datasets that contain organic, documented leaks** — nothing is injected or fabricated. For each, the detector flags the leaky column(s), then we quantify the leak's damage by training models **with** vs **without** the leak on identical splits (`python -m src.validate_detector`).

### Detection: every real leak was flagged

| Dataset | Target | Leaky column(s) | How it was caught |
|---|---|---|---|
| Parkinson's Disease | `PC` | `Duration` | **Hard degeneracy flag** (`Duration=0` for all controls) |
| Hotel Booking Demand | `is_canceled` | `reservation_status` | **Hard degeneracy flag** (constant within the not-cancelled class) |
| Bank Marketing | `y` | `duration` | Top-ranked suspicious (#1, AUC 0.93) — *provider-documented leak* |
| Heart Failure | `DEATH_EVENT` | `time` | Top-ranked suspicious (#1, AUC 0.84) |
| Cervical Cancer | `Biopsy` | `Schiller`, `Hinselmann`, `Citology` | Top-ranked suspicious (#1/#2/#3) |

### Impact: what each leak does to a model (Random Forest)

| Dataset | Leak type | Acc (with leak) | Acc (removed) | AUC (with leak) | AUC (removed) |
|---|---|---|---|---|---|
| Parkinson's Disease | within-class degeneracy | 1.000 | 0.800 | 1.000 | 0.807 |
| Bank Marketing | correlation (provider-documented) | 0.906 | 0.891 | 0.925 | 0.789 |
| Heart Failure | correlation (observation window) | 0.833 | 0.700 | 0.891 | 0.797 |
| Cervical Cancer | target leakage (alternate diagnoses) | 0.954 | 0.936 | 0.971 | 0.693 |
| Hotel Booking Demand | target leakage (label restated) | 1.000 | 0.894 | 1.000 | 0.959 |

*(Uniform default models are used here for cross-dataset comparability. Full, rigorously tuned per-dataset development — 7 models, grid search, CIs — lives in [`notebooks/`](notebooks/); e.g. the Hotel Booking case study independently rediscovers the leak before the detector confirms it.)*

**Key finding — AUC exposes leaks that accuracy hides.** On Cervical Cancer, removing the leak barely moves accuracy (0.954 → 0.936) because the dataset is imbalanced, yet **AUC collapses from 0.971 to 0.693**. A team watching only accuracy would never notice the leak. This is the central argument for a dedicated pre-training audit.

### Precision controls (clean data — no false alarms)
- **Breast Cancer Wisconsin:** legitimately strong features (`worst perimeter`, AUC > 0.97) rank high but trigger **zero** degeneracy hard-flags. ✅

### Honest scope: decision-support, not auto-deletion
The detector outputs a **ranked list for human review**, not automated verdicts — deliberately. On Hotel Booking the degeneracy flag also fired on `required_car_parking_spaces` (a genuine within-class degeneracy, but AUC only 0.55 — *not* a meaningful leak). Combined with its low predictiveness, a human dismisses it in seconds. The tool surfaces candidates; the analyst decides. It does **not** catch temporal, train/test-contamination, or multivariate leakage.

## 🛠️ Validating the Tool on the Original Inspiration (Parkinson's Dataset)

To demonstrate the value of the tool, the dataset from the original 2023 thesis was fed into the automated detector. As intended, the detector immediately hard-flagged the `Duration` column as a structural leak, automating the exact discovery that required manual investigation in 2023.

To quantify the danger of this specific leak if it *hadn't* been caught, we trained models both with and without the leaked column. 

The **"Honest" column reproduces the baseline of the 2023 thesis results**, where the leak was properly excluded. Note that while the fully hyperparameter-tuned models in the official thesis reported AUCs clustering tightly between **0.75 and 0.79**, the "Honest" metrics below use uniform, untuned default models for cross-dataset comparability (yielding a wider 0.588–0.798 AUC range). 

The **"Leaked" column demonstrates what a naive model produces if the detector is not used and the leak slips through** — artificially perfect 1.0 AUC scores.

| Model | Accuracy (Leaked) | Accuracy (Honest) | Accuracy Diff | AUC (Leaked) | AUC (Honest) |
|-------|-------------------|-------------------|---------------|--------------|--------------|
| Random Forest | 1.000 | 0.831 | -0.169 | 1.000 | 0.788 |
| XGBoost | 1.000 | 0.754 | -0.246 | 1.000 | 0.762 |
| Decision Tree | 1.000 | 0.754 | -0.246 | 1.000 | 0.747 |
| Logistic Reg. | 0.984 | 0.692 | -0.292 | 1.000 | 0.798 |
| SVC | 1.000 | 0.677 | -0.323 | 0.999 | 0.658 |
| KNN | 0.800 | 0.662 | -0.138 | 0.900 | 0.588 |
| MLP | 1.000 | 0.646 | -0.354 | 0.999 | 0.716 |

This realistic baseline validates the original 2023 thesis conclusions that differentiating healthy controls is challenging using purely biomechanical features, while simultaneously proving the necessity of automated Data QA tools in medical MLOps to prevent catastrophic false-confidence.

### Subgroup Fairness

Fairness was audited using the proxy feature `Dominant`. On the honest evaluation, Random Forest accuracy showed a noticeable performance gap across subgroups (0.889 vs 0.759). This indicates a potential statistical bias along handedness/laterality that requires further clinical investigation before deployment.
