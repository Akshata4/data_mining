import os, pandas as pd, joblib
from src.features.pipeline import build_preprocessor, TARGET
PROC="data/processed"; os.makedirs(PROC, exist_ok=True)
train=pd.read_csv(f"{PROC}/train.csv"); val=pd.read_csv(f"{PROC}/val.csv"); test=pd.read_csv(f"{PROC}/test.csv")
prep=build_preprocessor()
Xt=prep.fit_transform(train); Xv=prep.transform(val); Xs=prep.transform(test)
pd.DataFrame(Xt).to_parquet(f"{PROC}/train.parquet"); pd.DataFrame(Xv).to_parquet(f"{PROC}/val.parquet"); pd.DataFrame(Xs).to_parquet(f"{PROC}/test.parquet")
joblib.dump(prep, "models/preprocessor.joblib")
print("Modify complete.")
