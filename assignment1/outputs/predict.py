#!/usr/bin/env python3
import sys, json, pandas as pd, joblib, re
from pathlib import Path
def norm(c):
    c = str(c).strip().lower().replace("-", "_")
    c = re.sub(r"\s+", "_", c)
    return re.sub(r"[^0-9a-z_]", "", c)
if len(sys.argv) < 3:
    print("Usage: predict.py <input_csv> <output_csv>", file=sys.stderr); sys.exit(1)
in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
df = pd.read_csv(in_path)
df.columns = [norm(c) for c in df.columns]
df = df.drop(columns=[c for c in df.columns if c=='lung_cancer'], errors='ignore')
model = joblib.load(str(Path('outputs/lung_cancer_pipeline.joblib')))
probs = model.predict_proba(df)[:,1]
preds = (probs >= 0.558631).astype(int)
out = df.copy()
out['pred_prob'] = probs
out['pred_label'] = preds
out.to_csv(out_path, index=False)
print(f"Wrote predictions to {out_path}")
