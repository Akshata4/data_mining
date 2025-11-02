import os, numpy as np, pandas as pd, json
from sklearn.model_selection import GroupShuffleSplit
SEED=42; TARGET="Survived"; ID_COL="PassengerId"
RAW="data/raw"; PROC="data/processed"
os.makedirs(PROC, exist_ok=True)

def load_data():
    for fn in ["data/raw/train.csv","data/raw/titanic.csv","train.csv","titanic.csv"]:
        if os.path.exists(fn): return pd.read_csv(fn), False
    rng=np.random.default_rng(SEED); n=800
    pid=np.arange(1,n+1)
    pclass=rng.choice([1,2,3], n, p=[0.24,0.20,0.56])
    sex=rng.choice(["male","female"], n, p=[0.65,0.35])
    titles=np.where(sex=="female", rng.choice(["Mrs","Miss"], n, p=[0.55,0.45]),
                    rng.choice(["Mr","Master","Dr"], n, p=[0.88,0.10,0.02]))
    last=rng.choice(["Smith","Johnson","Brown","Taylor","Williams","Davies","Miller","Wilson","Moore","Clark"], n)
    first=rng.choice(["John","James","William","Charles","George","Henry","Arthur","Albert","Edward","Robert",
                      "Mary","Anna","Elizabeth","Margaret","Florence","Ethel"], n)
    name=[f"{l}, {t}. {f}" for l,t,f in zip(last,titles,first)]
    age=rng.normal(30,14,n).clip(0,80); age[rng.random(n)<0.2]=np.nan
    sibsp=rng.poisson(0.5,n).clip(0,8); parch=rng.poisson(0.4,n).clip(0,6)
    fare=np.round(np.exp(rng.normal(3.0,0.8,n))-5,2); fare[fare<0]=0.0
    embarked=rng.choice(["S","C","Q"], n, p=[0.72,0.18,0.10])
    cabin=np.array([None]*n, dtype=object); ticket=rng.integers(100000,999999,n).astype(str)
    prob=0.38 + 0.18*(sex=="female").astype(float) + 0.10*(pclass==1) - 0.06*(pclass==3)
    prob += np.where(~np.isnan(age)&(age<15),0.15,0.0); prob=np.clip(prob,0.02,0.98)
    y=rng.binomial(1,prob).astype(int)
    df=pd.DataFrame({"PassengerId":pid,"Survived":y,"Pclass":pclass,"Name":name,"Sex":sex,"Age":age,
                     "SibSp":sibsp,"Parch":parch,"Ticket":ticket,"Fare":fare,"Cabin":cabin,"Embarked":embarked})
    return df, True

df, demo=load_data()
df["Ticket_norm"]=(df["Ticket"].astype(str).str.replace(r"\s+","",regex=True)
                   .str.replace(r"[^A-Za-z0-9]","",regex=True).str.upper())
gss=GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
idx_tv, idx_te = next(gss.split(df, groups=df["Ticket_norm"]))
tv, te = df.iloc[idx_tv], df.iloc[idx_te]
gss2=GroupShuffleSplit(n_splits=1, test_size=0.15/(1-0.15), random_state=SEED)
idx_tr, idx_va = next(gss2.split(tv, groups=tv["Ticket_norm"]))
tr, va = tv.iloc[idx_tr], tv.iloc[idx_va]
assert set(tr["Ticket_norm"]).isdisjoint(va["Ticket_norm"]) and set(tr["Ticket_norm"]).isdisjoint(te["Ticket_norm"]) and set(va["Ticket_norm"]).isdisjoint(te["Ticket_norm"])
tr.to_csv(f"{PROC}/train.csv", index=False); va.to_csv(f"{PROC}/val.csv", index=False); te.to_csv(f"{PROC}/test.csv", index=False)
stats={"demo_mode": bool(demo), "splits":{
  "train":{"rows": int(len(tr)), "pos_rate": float(tr["Survived"].mean()), "groups": int(tr["Ticket_norm"].nunique())},
  "val":{"rows": int(len(va)), "pos_rate": float(va["Survived"].mean()), "groups": int(va["Ticket_norm"].nunique())},
  "test":{"rows": int(len(te)), "pos_rate": float(te["Survived"].mean()), "groups": int(te["Ticket_norm"].nunique())}}}
os.makedirs("reports", exist_ok=True)
with open("reports/partition_stats.json","w") as f: json.dump(stats, f, indent=2)
print("Sample complete.")
