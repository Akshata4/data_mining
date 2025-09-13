#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lung Cancer Risk Classification — End-to-End (CRISP-DM)
======================================================
- Reproducible (seed=42), leakage-proof pipelines (ColumnTransformer)
- Stratified 70/15/15 split
- Baselines: Logistic Regression, Decision Tree
- Stronger models: RandomForest + XGBoost/LightGBM fallback
- Threshold selection: Recall≥0.85 (max precision) else Youden’s J
- Explainability: permutation importance, PDP for age/pack_years
- Final test evaluation; calibration; artifacts saved to --outdir

USAGE (basic):
  python lung_cancer_risk_end_to_end.py --data lung_cancer_dataset.csv

USAGE (custom outdir, skip plots):
  python lung_cancer_risk_end_to_end.py --data lung_cancer_dataset.csv --outdir results --no-plots

Requires: pandas, numpy, scikit-learn, matplotlib; optional: xgboost or lightgbm, scipy
"""

import argparse, os, re, sys, json, math, warnings, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, auc, confusion_matrix, brier_score_loss
)
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.calibration import calibration_curve

# Try SciPy utilities if available (optional)
try:
    from scipy.stats import pointbiserialr, chi2_contingency
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

warnings.filterwarnings("ignore")
np.random.seed(42); random.seed(42)

# ------------------------
# Utilities
# ------------------------
def normalize_col(c: str) -> str:
    c = str(c).strip()
    c = re.sub(r"\s+", "_", c).replace("-", "_")
    c = re.sub(r"[^0-9a-zA-Z_]", "", c).strip("_").lower()
    return c

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig_safe(path: Path, do_plots: bool):
    if do_plots:
        plt.tight_layout()
        plt.savefig(path)
        plt.show()
    else:
        plt.close()

def summarize_split(yarr):
    vc = pd.Series(yarr).value_counts()
    return {"n": int(vc.sum()), "pos": int(vc.get(1,0)), "neg": int(vc.get(0,0)), "pos_rate": float((vc.get(1,0)/max(1,vc.sum())))}

def make_preprocessor(num_cols, cat_cols):
    num_pipe = SkPipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_pipe = SkPipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])

def evaluate_probs(y_true, probs, threshold=0.5):
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall + 1e-12) if (precision + recall) > 0 else 0.0
    return {
        "AUROC": roc_auc_score(y_true, probs),
        "PR_AUC(AP)": average_precision_score(y_true, probs),
        "Accuracy": (tp + tn) / cm.sum(),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "threshold": float(threshold)
    }

def choose_threshold(y_true, probs, recall_target=0.85):
    fpr, tpr, roc_thr = roc_curve(y_true, probs)
    prec, rec, pr_thr = precision_recall_curve(y_true, probs)
    # Recall-first policy
    mask = rec[:-1] >= recall_target
    if mask.any():
        idxs = np.where(mask)[0]
        best_idx = idxs[np.argmax(prec[:-1][idxs])]
        return float(pr_thr[best_idx]), "Recall≥{:.2f} (max precision)".format(recall_target)
    # Fallback: Youden's J
    j_idx = int(np.argmax(tpr - fpr))
    thr = float(roc_thr[j_idx]) if j_idx < len(roc_thr) else 0.5
    return thr, "Youden's J (fallback)"

# ------------------------
# Steps
# ------------------------
def load_and_eda(data_path: Path, outdir: Path, do_plots: bool):
    print("\n[Step 2] Data Understanding & EDA")
    df_raw = pd.read_csv(data_path)
    df = df_raw.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    assert 'lung_cancer' in df.columns, "Target column 'lung_cancer' not found."
    # Target 0/1
    y = df['lung_cancer'].astype(str).str.strip().str.lower().map(
        {'yes':1,'y':1,'true':1,'1':1,'no':0,'n':0,'false':0,'0':0}
    ).astype(int).values
    # Schema
    schema = []
    for c in df.columns:
        schema.append((c, str(df[c].dtype), int(df[c].nunique(dropna=True)), int(df[c].isna().sum()),
                       round(100*df[c].isna().mean(),2)))
    schema_df = pd.DataFrame(schema, columns=['column','dtype','unique_values','missing','missing_%']).sort_values('column')
    schema_df.to_csv(outdir/"eda_schema.csv", index=False)
    # Target distribution plot
    tvc = pd.Series(y).value_counts().sort_index()
    plt.figure()
    plt.bar(['No (0)','Yes (1)'], [tvc.get(0,0), tvc.get(1,0)])
    plt.title('Target distribution: lung_cancer'); plt.xlabel('Class'); plt.ylabel('Count')
    savefig_safe(outdir/"fig_target_distribution.png", do_plots)
    print("Positive rate:", f"{tvc.get(1,0)/max(1,tvc.sum()):.1%}")
    # ID-like columns
    n_rows = len(df)
    id_like = []
    for c in df.columns:
        if c == 'lung_cancer':
            continue
        if df[c].nunique(dropna=True)/n_rows > 0.95 or re.search(r'\bid\b', c):
            id_like.append(c)
    print("ID-like columns:", id_like if id_like else "None")
    # Numeric/categorical for EDA
    num_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in id_like]
    cat_cols = [c for c in df.columns if c not in num_cols + id_like + ['lung_cancer']]
    # Numeric histograms + summary
    rows = []
    for c in num_cols:
        s = df[c]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
        rows.append((c, float(s.min()), float(s.max()), float(s.mean()), float(s.median()), float(s.std()), outliers))
        plt.figure(); plt.hist(s.dropna(), bins=30)
        plt.title(f'Histogram: {c}'); plt.xlabel(c); plt.ylabel('Frequency')
        savefig_safe(outdir/f"fig_hist_{c}.png", do_plots)
    pd.DataFrame(rows, columns=['feature','min','max','mean','median','std','approx_outliers(IQR)']).to_csv(outdir/"eda_numeric_summary.csv", index=False)
    # Categorical levels (top 8)
    rows = []
    for c in cat_cols:
        vc = df[c].astype(str).value_counts(dropna=False).head(8)
        for k,v in vc.items():
            rows.append((c, str(k), int(v), float(v/len(df))))
    pd.DataFrame(rows, columns=['feature','level','count','fraction']).to_csv(outdir/"eda_categorical_top_levels.csv", index=False)
    # Numeric ↔ target (point-biserial) + boxplots
    if SCIPY_OK:
        rows = []
        for c in num_cols:
            s = pd.to_numeric(df[c], errors='coerce')
            m = ~s.isna()
            r, p = pointbiserialr(y[m], s[m]) if m.sum() else (np.nan, np.nan)
            rows.append((c, float(r) if r==r else np.nan, float(p) if p==p else np.nan))
            plt.figure(); plt.boxplot([s[y==0].dropna(), s[y==1].dropna()], labels=['No(0)','Yes(1)'])
            plt.title(f'{c} by target'); plt.ylabel(c)
            savefig_safe(outdir/f"fig_box_{c}_by_target.png", do_plots)
        pd.DataFrame(rows, columns=['numeric_feature','point_biserial_r','p_value']).to_csv(outdir/"eda_num_target_correlation.csv", index=False)
    # Categorical ↔ target (pos rate + Cramér’s V)
    if SCIPY_OK:
        rate_rows, cramer_rows = [], []
        yy = pd.Series(y, name='y')
        for c in cat_cols:
            grp = df.groupby(c)['lung_cancer'].apply(lambda s: (s.astype(str).str.lower().isin(['yes','y','true','1'])).mean())
            cnt = df[c].value_counts()
            tmp = pd.DataFrame({'pos_rate': grp, 'count': cnt}).reset_index().sort_values('count', ascending=False).head(12)
            for _, r in tmp.iterrows():
                rate_rows.append((c, str(r[c]), float(r['pos_rate']), int(r['count'])))
            ct = pd.crosstab(df[c], yy)
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, p, dof, _ = chi2_contingency(ct, correction=False)
                n = ct.values.sum(); k = min(ct.shape)-1
                if n>0 and k>0:
                    v = math.sqrt(chi2/(n*k)); cramer_rows.append((c, float(v), float(p), int(ct.shape[0])))
        pd.DataFrame(rate_rows, columns=['feature','level','positive_rate','count']).to_csv(outdir/"eda_cat_positive_rates.csv", index=False)
        pd.DataFrame(cramer_rows, columns=['categorical_feature','cramers_v','p_value','n_levels']).sort_values('cramers_v', ascending=False).to_csv(outdir/"eda_cat_cramers_v.csv", index=False)
    return df, y, id_like, num_cols, cat_cols

def prepare_data(df: pd.DataFrame, y, id_like, outdir: Path):
    print("\n[Step 3] Data Preparation")
    X = df.drop(columns=['lung_cancer'] + id_like, errors='ignore')
    # Feature typing
    force_numeric = [c for c in ['age','pack_years'] if c in X.columns]
    num_candidates = X.select_dtypes(include=['number']).columns.tolist()
    low_card_num_as_cat = [c for c in num_candidates if c not in force_numeric and X[c].nunique(dropna=True) <= 10]
    num_features = sorted(set(force_numeric + [c for c in num_candidates if c not in low_card_num_as_cat]))
    cat_features = sorted(set([c for c in X.columns if c not in num_candidates] + low_card_num_as_cat))
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    valid_rel = 0.15 / 0.85
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_rel, stratify=y_train, random_state=42)
    # Save split summary
    split_info = pd.DataFrame([
        {"split":"train", **summarize_split(y_train)},
        {"split":"valid", **summarize_split(y_valid)},
        {"split":"test",  **summarize_split(y_test)},
    ])
    split_info.to_csv(outdir/"split_summary.csv", index=False)
    print(split_info)
    return X_train, X_valid, X_test, y_train, y_valid, y_test, num_features, cat_features

def fit_baselines(X_train, y_train, X_valid, y_valid, num_features, cat_features):
    print("\n[Step 4] Baseline Models")
    pre = make_preprocessor(num_features, cat_features)
    logreg = SkPipeline([('prep', pre), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])
    dtree  = SkPipeline([('prep', pre), ('clf', DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight='balanced', random_state=42))])
    logreg.fit(X_train, y_train); dtree.fit(X_train, y_train)
    p_lr = logreg.predict_proba(X_valid)[:,1]; p_dt = dtree.predict_proba(X_valid)[:,1]
    res = pd.DataFrame([
        {"model":"LogisticRegression", **evaluate_probs(y_valid, p_lr)},
        {"model":"DecisionTree", **evaluate_probs(y_valid, p_dt)},
    ]).sort_values(['AUROC','PR_AUC(AP)'], ascending=False).reset_index(drop=True)
    print(res[['model','AUROC','PR_AUC(AP)','Accuracy','Precision','Recall','F1']])
    return res, logreg, dtree

def fit_strong_models(X_train, y_train, X_valid, y_valid, num_features, cat_features):
    print("\n[Step 5] Stronger Models")
    pre = make_preprocessor(num_features, cat_features)
    rf = SkPipeline([('prep', pre), ('clf', RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5, min_samples_split=10,
        max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=42
    ))])
    rf.fit(X_train, y_train)
    boost_name = None
    # Try XGBoost -> LightGBM -> GradientBoosting
    try:
        from xgboost import XGBClassifier
        scale_pos_weight = (np.sum(y_train==0)/max(1,np.sum(y_train==1)))
        boost = SkPipeline([('prep', pre), ('clf', XGBClassifier(
            objective='binary:logistic', eval_metric='auc', tree_method='hist',
            n_estimators=400, learning_rate=0.08, max_depth=5,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            min_child_weight=2.0, gamma=0.0, scale_pos_weight=scale_pos_weight, random_state=42, verbosity=0
        ))])
        boost.fit(X_train, y_train); boost_name = "XGBoost"
    except Exception:
        try:
            from lightgbm import LGBMClassifier
            boost = SkPipeline([('prep', pre), ('clf', LGBMClassifier(
                objective='binary', n_estimators=500, learning_rate=0.06, num_leaves=64,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, is_unbalance=True,
                random_state=42, n_jobs=-1
            ))])
            boost.fit(X_train, y_train); boost_name = "LightGBM"
        except Exception:
            boost = SkPipeline([('prep', pre), ('clf', GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.08, max_depth=3, subsample=0.9, random_state=42
            ))])
            boost.fit(X_train, y_train); boost_name = "GradientBoosting"
    # Evaluate
    p_rf = rf.predict_proba(X_valid)[:,1]
    p_bo = boost.predict_proba(X_valid)[:,1]
    res = pd.DataFrame([
        {"model":"RandomForest", **evaluate_probs(y_valid, p_rf)},
        {"model":boost_name, **evaluate_probs(y_valid, p_bo)},
    ]).sort_values(['AUROC','PR_AUC(AP)'], ascending=False).reset_index(drop=True)
    print(res[['model','AUROC','PR_AUC(AP)','Accuracy','Precision','Recall','F1']])
    # Pick winner
    winner_name = res.iloc[0]['model']
    winner = rf if winner_name == "RandomForest" else boost
    return res, winner_name, winner

def threshold_and_explain(winner, X_valid, y_valid, outdir: Path, do_plots: bool):
    import numpy as np, pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    from sklearn.inspection import permutation_importance, partial_dependence

    print("\n[Step 6-7] Threshold Selection & Explainability")
    pv = winner.predict_proba(X_valid)[:, 1]

    # --- Threshold (recall-first, else Youden's J) ---
    fpr, tpr, roc_thr = roc_curve(y_valid, pv)
    prec, rec, pr_thr = precision_recall_curve(y_valid, pv)
    mask = rec[:-1] >= 0.85
    if mask.any():
        idxs = np.where(mask)[0]
        best_idx = idxs[np.argmax(prec[:-1][idxs])]
        thr = float(pr_thr[best_idx]); policy = "Recall≥0.85 (max precision)"
    else:
        j_idx = int(np.argmax(tpr - fpr))
        thr = float(roc_thr[j_idx]) if j_idx < len(roc_thr) else 0.5
        policy = "Youden's J (fallback)"

    print(f"Threshold policy: {policy} | chosen threshold: {thr:.3f}")

    # --- Curves (only if plotting) ---
    if do_plots:
        plt.figure(); plt.plot(fpr, tpr, label=f'ROC (AUC={auc(fpr,tpr):.3f})'); plt.plot([0,1],[0,1],'--')
        plt.title('ROC (validation)'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()
        savefig_safe(outdir/"fig_roc_validation.png", True)

        plt.figure(); plt.plot(rec, prec, label='PR')
        plt.title('Precision-Recall (validation)'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()
        savefig_safe(outdir/"fig_pr_validation.png", True)

    # --- Threshold sweep CSV ---
    ths = np.linspace(0.05, 0.95, 19)
    rows = []
    from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
    for t in ths:
        yb = (pv >= t).astype(int)
        cm = confusion_matrix(y_valid, yb)
        tn, fp, fn, tp = cm.ravel()
        prec_t = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_t  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_t   = (2*prec_t*rec_t) / (prec_t+rec_t+1e-12) if (prec_t+rec_t) > 0 else 0.0
        rows.append(dict(threshold=float(t), precision=prec_t, recall=rec_t, f1=f1_t, TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn)))
    pd.DataFrame(rows).to_csv(outdir/"threshold_sweep_validation.csv", index=False)

    # --- Permutation importance (model-agnostic) ---
    pi = permutation_importance(winner, X_valid, y_valid, n_repeats=5, random_state=42, scoring='roc_auc')
    pi_df = pd.DataFrame({
        'feature': X_valid.columns,
        'importance_mean': pi.importances_mean,
        'importance_std':  pi.importances_std
    }).sort_values('importance_mean', ascending=False)
    pi_df.to_csv(outdir/"permutation_importance_validation.csv", index=False)

    # --- PDPs (plot only; handle sklearn>=1.4 and older) ---
    if do_plots:
        for feat in ['pack_years', 'age']:
            if feat in X_valid.columns:
                try:
                    # Prefer the modern API for plotting
                    from sklearn.inspection import PartialDependenceDisplay
                    PartialDependenceDisplay.from_estimator(
                        winner, X_valid, [feat], kind='average', grid_resolution=20
                    )
                    plt.title(f'Partial Dependence: {feat}')
                    savefig_safe(outdir/f"fig_pdp_{feat}.png", True)
                except Exception:
                    # Fallback: extract arrays from Bunch (grid_values in new, values in old)
                    pdp = partial_dependence(winner, X_valid, [feat], kind='average', grid_resolution=20)
                    xs = pdp.get('grid_values', pdp.get('values'))[0]
                    avg = pdp['average']
                    ys = np.asarray(avg).ravel()
                    plt.figure(); plt.plot(np.asarray(xs).ravel(), ys)
                    plt.title(f'Partial Dependence: {feat}')
                    plt.xlabel(feat); plt.ylabel('Partial dependence')
                    savefig_safe(outdir/f"fig_pdp_{feat}.png", True)

    return thr, policy


def finalize_and_test(winner, thr, policy, X_train, X_valid, X_test, y_train, y_valid, y_test, outdir: Path, dropped_ids, num_features, cat_features, winner_name):
    print("\n[Step 8] Finalize & Test")
    X_trv = pd.concat([X_train, X_valid]).reset_index(drop=True)
    y_trv = np.concatenate([y_train, y_valid])
    final_model = SkPipeline(winner.steps)
    final_model.fit(X_trv, y_trv)
    pt = final_model.predict_proba(X_test)[:,1]
    # Metrics
    auroc = roc_auc_score(y_test, pt)
    ap = average_precision_score(y_test, pt)
    brier = brier_score_loss(y_test, pt)
    test_metrics = evaluate_probs(y_test, pt, threshold=thr)
    test_metrics.update({"AUROC": auroc, "PR_AUC(AP)": ap, "Brier": brier})
    pd.DataFrame([test_metrics]).to_csv(outdir/"final_test_metrics.csv", index=False)
    print(pd.DataFrame([test_metrics])[['AUROC','PR_AUC(AP)','Brier','Accuracy','Precision','Recall','F1','TP','FP','FN','TN','threshold']])
    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, pt, n_bins=10, strategy='uniform')
    plt.figure(); plt.plot([0,1],[0,1],'--',label='Perfect'); plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.title('Calibration curve (test)'); plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction of positives'); plt.legend()
    savefig_safe(outdir/"fig_calibration_test.png", True)
    # Artifacts
    import joblib, sklearn, numpy, matplotlib
    model_path = outdir/"lung_cancer_pipeline.joblib"
    meta_path  = outdir/"model_meta.json"
    predict_py = outdir/"predict.py"
    req_path   = outdir/"requirements.txt"
    # Save model
    joblib.dump(final_model, model_path)
    # Metadata
    meta = {
        "framework": "scikit-learn",
        "winner_model": winner_name,
        "random_state": 42,
        "threshold_policy": policy,
        "threshold": float(thr),
        "dropped_id_cols": dropped_ids,
        "numeric_features": list(num_features),
        "categorical_features": list(cat_features),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__
        }
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    # Predict script
    predict_src = f"""#!/usr/bin/env python3
import sys, json, pandas as pd, joblib, re
from pathlib import Path
def norm(c):
    c = str(c).strip().lower().replace("-", "_")
    c = re.sub(r"\\s+", "_", c)
    return re.sub(r"[^0-9a-z_]", "", c)
if len(sys.argv) < 3:
    print("Usage: predict.py <input_csv> <output_csv>", file=sys.stderr); sys.exit(1)
in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
df = pd.read_csv(in_path)
df.columns = [norm(c) for c in df.columns]
df = df.drop(columns=[c for c in df.columns if c=='lung_cancer'], errors='ignore')
model = joblib.load(str(Path('{model_path}')))
probs = model.predict_proba(df)[:,1]
preds = (probs >= {float(thr):.6f}).astype(int)
out = df.copy()
out['pred_prob'] = probs
out['pred_label'] = preds
out.to_csv(out_path, index=False)
print(f"Wrote predictions to {{out_path}}")
"""
    predict_path = Path(predict_py)
    predict_path.write_text(predict_src)
    # Requirements
    reqs = [f"python=={meta['versions']['python']}", f"numpy=={meta['versions']['numpy']}", f"pandas=={meta['versions']['pandas']}",
            f"scikit-learn=={meta['versions']['scikit-learn']}", f"joblib=={meta['versions']['joblib']}", f"matplotlib=={meta['versions']['matplotlib']}"]
    Path(req_path).write_text("\n".join(reqs))
    print("\nArtifacts saved:")
    print(" -", model_path)
    print(" -", meta_path)
    print(" -", predict_py)
    print(" -", req_path)

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Lung cancer risk classification (CRISP-DM end-to-end)")
    parser.add_argument("--data", required=True, type=str, help="Path to lung_cancer_dataset.csv")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to write figures & artifacts")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting (faster)")
    args = parser.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print("[Step 1] Business Understanding")
    print(" - Goal: binary classification of lung cancer risk; emphasis on high recall screening.")
    print(" - Primary metric: AUROC; secondary: PR-AUC; threshold chosen with Recall≥0.85 policy.")

    # Step 2: EDA
    df, y, id_like, num_cols, cat_cols = load_and_eda(data_path, outdir, do_plots=(not args.no_plots))

    # Step 3: Prep
    X_train, X_valid, X_test, y_train, y_valid, y_test, num_features, cat_features = prepare_data(df, y, id_like, outdir)

    # Step 4: Baselines
    res_base, logreg, dtree = fit_baselines(X_train, y_train, X_valid, y_valid, num_features, cat_features)

    # Step 5: Stronger models
    res_strong, winner_name, winner = fit_strong_models(X_train, y_train, X_valid, y_valid, num_features, cat_features)

    # Step 6-7: Threshold + Explainability
    thr, policy = threshold_and_explain(winner, X_valid, y_valid, outdir, do_plots=(not args.no_plots))

    # Step 8: Finalize & Test + Artifacts
    finalize_and_test(winner, thr, policy, X_train, X_valid, X_test, y_train, y_valid, y_test, outdir,
                      id_like, num_features, cat_features, winner_name)

    print("\nDone. Inspect outputs at:", outdir.resolve())

if __name__ == "__main__":
    main()
