import os, json, numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, accuracy_score, brier_score_loss, confusion_matrix)

TARGET="Survived"; PROC="data/processed"; os.makedirs("reports", exist_ok=True)
train=pd.read_csv(f"{PROC}/train.csv"); val=pd.read_csv(f"{PROC}/val.csv"); test=pd.read_csv(f"{PROC}/test.csv")

# model: use trained best if available; else LR(bal)
try:
    model=joblib.load("models/logreg_bal.joblib")
except Exception:
    from src.features.pipeline import build_preprocessor
    from sklearn.linear_model import LogisticRegression
    prep=build_preprocessor()
    model=Pipeline([("prep",prep),("model",LogisticRegression(max_iter=700, class_weight="balanced"))]).fit(train, train[TARGET])

y_val=val[TARGET].values
cal_sig=CalibratedClassifierCV(model, method="sigmoid", cv="prefit").fit(val, y_val)
cal_iso=CalibratedClassifierCV(model, method="isotonic", cv="prefit").fit(val, y_val)
p_sig=cal_sig.predict_proba(val)[:,1]; p_iso=cal_iso.predict_proba(val)[:,1]
b_sig=brier_score_loss(y_val,p_sig); b_iso=brier_score_loss(y_val,p_iso)
cal=cal_iso if b_iso<b_sig else cal_sig; cal_name="isotonic" if b_iso<b_sig else "sigmoid"

ratios=[(1,1),(1,2),(1,3),(1,5)]; ths=np.linspace(0.05,0.95,19); p_val=cal.predict_proba(val)[:,1]
def cost(y,p,t,cfp=1,cfn=2):
    yhat=(p>=t).astype(int); fp=((yhat==1)&(y==0)).sum(); fn=((yhat==0)&(y==1)).sum(); return cfp*fp + cfn*fn
thr={f"{cfp}:{cfn}": float(ths[np.argmin([cost(y_val,p_val,t,cfp,cfn) for t in ths])]) for (cfp,cfn) in ratios}
thr_star=thr["1:2"]

y_test=test[TARGET].values; p_test=cal.predict_proba(test)[:,1]
rng=np.random.default_rng(123); idx=np.arange(len(y_test))
def boot_ci(fn, B=300):
    vals=[ fn(*(lambda s:(y_test[s],p_test[s]))(rng.choice(idx,size=len(idx),replace=True)) ) for _ in range(B) ]
    return float(np.mean(vals)), float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))
roc=boot_ci(lambda y,p: roc_auc_score(y,p)); pr=boot_ci(lambda y,p: average_precision_score(y,p))
f1=boot_ci(lambda y,p: f1_score(y,(p>=thr_star).astype(int)))
acc=boot_ci(lambda y,p: accuracy_score(y,(p>=thr_star).astype(int)))
brier=boot_ci(lambda y,p: brier_score_loss(y,p))
def ece10(y,p):
    bins=np.linspace(0,1,11); ids=np.digitize(p,bins)-1; e=0.0
    for b in range(10):
        m=ids==b
        if np.any(m): e += abs((y[m].mean()-p[m].mean()))*(m.mean())
    return e
ece=boot_ci(ece10)
cm=confusion_matrix(y_test, (p_test>=thr_star).astype(int)).tolist()
out={"calibrator":cal_name,"thr_grid":thr,"thr_star_FNFP_1_2":float(thr_star),
     "test_metrics":{"roc_auc":roc,"pr_auc":pr,"f1_thr":f1,"acc_thr":acc,"brier":brier,"ece10":ece},
     "confusion_matrix":cm}
with open("reports/holdout_metrics.json","w") as f: json.dump(out, f, indent=2)
print("Assess complete.")
