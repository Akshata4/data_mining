import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
PROC="data/processed"; os.makedirs("reports", exist_ok=True); os.makedirs("assets", exist_ok=True)
train=pd.read_csv(f"{PROC}/train.csv")

miss=train.isna().mean().sort_values(ascending=False)
miss.to_csv("reports/missingness.csv")
plt.figure(); miss.plot(kind="bar"); plt.title("Missingness (Train)"); plt.tight_layout(); plt.savefig("assets/missingness.png")

num=train.select_dtypes(include=[np.number]).drop(columns=["PassengerId"], errors="ignore")
corr=num.corr(method="pearson"); corr.to_csv("reports/corr_pearson.csv")
plt.figure(); plt.imshow(corr, vmin=-1, vmax=1); plt.title("Pearson Correlation"); plt.colorbar(); plt.tight_layout(); plt.savefig("assets/corr_heatmap.png")

def cramers_v(x,y):
    ct=pd.crosstab(x,y); chi2=stats.chi2_contingency(ct)[0]; n=ct.values.sum(); phi2=chi2/n; r,k=ct.shape
    phi2corr=max(0, phi2 - (k-1)*(r-1)/(n-1)); rcorr = r - (r-1)**2/(n-1); kcorr = k - (k-1)**2/(n-1)
    return float(np.sqrt(phi2corr / max(1e-9, min((kcorr-1),(rcorr-1)))))

rows=[]
for c in train.select_dtypes(exclude=[np.number]).columns:
    try: rows.append({"feature":c, "cramers_v":cramers_v(train[c].astype(str), train["Survived"])})
    except Exception: pass
pd.DataFrame(rows).to_csv("reports/cramers_v.csv", index=False)
pd.Series({"note":"Review Cabin/Name/Ticket patterns for leakage; consider subgroup calibration."}).to_csv("reports/leakage_notes.csv")
print("Explore complete.")
