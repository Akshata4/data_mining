# Lung Cancer Risk Classification — End-to-End (CRISP-DM)

A fully reproducible, leakage-proof **binary classification** pipeline that predicts **lung cancer risk** from tabular data using **scikit-learn**. It follows the **CRISP-DM** framework and emphasizes a **recall-first** thresholding policy suitable for screening scenarios.

> ⚕️ **Disclaimer:** Educational/ML workflow only — **not medical advice**.

---

## ✨ What you get

- **One command** to run the entire project end-to-end (EDA → prep → modeling → evaluation → artifacts).
- **Leakage-proof pipelines** (`ColumnTransformer` inside `Pipeline`).
- **Baseline models:** Logistic Regression, Decision Tree.
- **Strong models:** RandomForest + Boosted model (XGBoost preferred; LightGBM/GradientBoosting fallback).
- **Threshold selection:** choose the **highest-precision** threshold **with Recall ≥ 0.85** (fallback: Youden’s J).
- **Explainability:** permutation importance, PDPs for `age` & `pack_years` (when plots enabled).
- **Final artifacts:** saved `Pipeline` (`.joblib`), `predict.py`, `model_meta.json`, `requirements.txt`.
- **Reproducibility:** `random_state = 42` everywhere; **stratified** 70/15/15 split.



---

## 📦 Requirements

Use Python **3.10–3.12**. Minimal runtime deps:
* numpy
* pandas
* scikit-learn
* matplotlib
* joblib
---

## 🚀 Quickstart

Put your CSV next to the script and name it lung_cancer_dataset.csv (or pass a different path).

Run:

python main.py --data lung_cancer_dataset.csv


Useful flags:

* --outdir RESULTS — change output directory (default: outputs)

* --no-plots — skip plotting (faster / avoids backend or PDP issues)

---

## 📥 Input expectations

Target column lung_cancer with values like yes/no, y/n, true/false, 1/0 (case/whitespace handled).

Numeric features may include age, pack_years.

Categorical domain factors (when present) such as copd_diagnosis, asbestos_exposure, radon_exposure, family_history, secondhand_smoke_exposure.

Any ID-like columns (e.g., patient_id or *id*) are automatically dropped to prevent leakage.

## 🔬 What the script does (step-by-step)

### 1) Business Understanding
- Optimize for **high recall** in a screening context.
- Choose decision threshold by **Recall ≥ 0.85 (maximize precision)**; if not achievable, fall back to **Youden’s J**.

### 2) Data Understanding (EDA)
- Normalize column names; **assert target** (`lung_cancer`).
- Inspect **schema** & **missingness**; check **target balance**.
- Plot **numeric histograms** and scan **outliers**; summarize **categorical levels**.
- Compute **Numeric ↔ target** (point-biserial) and **Categorical ↔ target** (Cramér’s V) **if SciPy is available**.

### 3) Data Preparation
- **Drop IDs** (e.g., `patient_id`); keep **`age`** & **`pack_years`** as numeric; reclassify **low-cardinality numerics** as categorical.
- **Stratified 70/15/15 split** → train / validation / test (seed=42).
- Preprocessing via **ColumnTransformer**:
  - **Numeric:** median impute → standardize
  - **Categorical:** most-frequent impute → one-hot encode (handle_unknown='ignore')

### 4) Modeling
- **Baselines:** Logistic Regression, Decision Tree (both **class-balanced**).
- **Strong models:** RandomForest + **Boosted** (XGBoost preferred; fallback to LightGBM / GradientBoosting).
- Pick **validation winner** by **AUROC** (tie-break with **PR-AUC**).

### 5) Evaluation & Threshold
- Plot **ROC** / **PR** curves (if plots enabled).
- Apply **recall-first threshold selection**; save **threshold sweep CSV**.
- Report **confusion matrix** & **point metrics** on **validation**.

### 6) Explainability / Error Analysis
- **Permutation importance** on validation (ROC-AUC drop).
- **PDP** for **`age`** / **`pack_years`** (if present & plots enabled).

### 7) Finalize & Test
- **Retrain winner** on **train + valid**.
- **Evaluate once** on the **held-out test** at the chosen threshold.
- Plot **calibration curve** and compute **Brier score** (if plots enabled).

### 8) Artifacts
- `lung_cancer_pipeline.joblib` — ready-to-score **Pipeline** (preprocessing + model).
- `predict.py` — **batch scorer** CLI.
- `model_meta.json` — winner model, threshold/policy, features, versions.
- `requirements.txt` — exact versions logged.

---

## 📊 Interpreting outputs

- `final_test_metrics.csv` — **AUROC, PR-AUC, Brier, Accuracy, Precision, Recall, F1**, confusion-matrix counts, and the **threshold** used.
- `threshold_sweep_validation.csv` — **precision/recall/F1** across thresholds to visualize trade-offs.
- `permutation_importance_validation.csv` — top features by **ROC-AUC drop** when shuffled.

**Figures** *(when not using `--no-plots`)*:
- `fig_roc_validation.png`, `fig_pr_validation.png`, `fig_calibration_test.png`
- `fig_pdp_age.png`, `fig_pdp_pack_years.png` *(if features present)*

## 🧪 Batch scoring with the saved model

After a successful run:

`python outputs/predict.py new_patients.csv scored.csv`


Adds:

pred_prob — predicted probability of risk

pred_label — 0/1 using the saved threshold (see model_meta.json)

## 🔁 Reproducibility

`random_state = 42` across splits and models.

Stratified 70/15/15 ensures class balance in train/valid/test.

All preprocessing is inside the pipeline to avoid leakage.

## 🛠️ Troubleshooting

### 1) `KeyError: 'values'` during PDP / scikit-learn compatibility

Recent scikit-learn versions changed `partial_dependence` internals. Two fixes:

- **Easiest:** run with `--no-plots` (skips PDPs).
- **Recommended patch:** replace `threshold_and_explain(...)` with a version that uses  
  `PartialDependenceDisplay.from_estimator` and falls back to `grid_values` / `values`.  
  *(The provided script will still complete with `--no-plots`.)*

## 📈 Metrics & threshold policy (why Recall first?)

In screening contexts, **missing positives** can be costlier than investigating **false alarms**. We therefore:

1. Find thresholds where **Recall ≥ 0.85** on validation.  
2. Pick the one with **max Precision** among them.  
3. If unattainable, fall back to **Youden’s J** (maximizes **TPR – FPR**).

This policy is encoded and logged in `model_meta.json`.
