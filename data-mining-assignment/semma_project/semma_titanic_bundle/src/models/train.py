import os, numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.features.pipeline import build_preprocessor, TARGET

PROC="data/processed"; os.makedirs("reports", exist_ok=True)
train=pd.read_csv(f"{PROC}/train.csv"); groups=train.get("Ticket_norm", train["PassengerId"])
prep=build_preprocessor()
models={"logreg_bal": LogisticRegression(max_iter=700, class_weight="balanced", solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
        "gbdt": GradientBoostingClassifier(random_state=42, n_estimators=250, learning_rate=0.08, max_depth=3)}
X=train.copy(); y=train[TARGET].values
cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
rows=[]
for name, est in models.items():
    oof=np.zeros(len(X)); seen=np.zeros(len(X),dtype=bool)
    for tr,te in cv.split(X,y,groups=groups):
        pipe=Pipeline([("prep",prep),("model",est)])
        pipe.fit(X.iloc[tr], y[tr])
        oof[te]=pipe.predict_proba(X.iloc[te])[:,1]; seen[te]=True
    rows.append({"model":name,"roc_auc_oof":float(roc_auc_score(y[seen],oof[seen])),
                 "pr_auc_oof":float(average_precision_score(y[seen],oof[seen]))})
cv_df=pd.DataFrame(rows).sort_values("roc_auc_oof", ascending=False)
cv_df.to_csv("reports/cv_results.csv", index=False)
best=cv_df.iloc[0]["model"]
final=Pipeline([("prep",prep),("model",models[best])]).fit(X,y)
joblib.dump(final, f"models/{best}.joblib")
print("Model complete. Best:", best)
