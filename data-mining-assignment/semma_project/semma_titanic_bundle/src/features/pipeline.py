import re, numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

ID_COL = "PassengerId"
TARGET = "Survived"

def canon_title(t):
    t=str(t).strip().upper()
    mp={'MRS':'Mrs','MISS':'Miss','MS':'Miss','MLLE':'Miss','MME':'Mrs','MR':'Mr','MASTER':'Master',
        'DR':'Officer','REV':'Officer','COL':'Officer','MAJOR':'Officer','CAPT':'Officer',
        'SIR':'Royalty','LADY':'Royalty','COUNTESS':'Royalty','DON':'Royalty'}
    return mp.get(t, t.title() if t in {'MR','MRS','MISS','MASTER'} else 'Other')

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        pat = re.compile(r",\s*([A-Za-z]+)\.")
        X["Title"]=X["Name"].map(lambda s: canon_title(pat.search(str(s)).group(1) if pat.search(str(s)) else "NA"))
        deck = X["Cabin"].astype(str).str[0]
        deck = deck.mask(X["Cabin"].isna(), "U")
        X["Deck"] = deck
        X["FamilySize"] = X["SibSp"].fillna(0) + X["Parch"].fillna(0) + 1
        X["IsAlone"] = (X["FamilySize"]==1).astype(int)
        fs = X["FamilySize"].replace(0,1)
        X["FarePerPerson"] = (X["Fare"].fillna(0)/fs).replace([np.inf,-np.inf], np.nan)
        X["Pclass_Sex"] = "P"+X["Pclass"].astype(str)+"_"+X["Sex"].astype(str)
        keep = ["Age","Fare","SibSp","Parch","FamilySize","IsAlone","FarePerPerson",
                "Sex","Embarked","Pclass","Title","Deck","Pclass_Sex",
                ID_COL,"Name","Ticket","Ticket_norm","Cabin"]
        return X[keep]

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, cols, min_count=8, min_prop=0.01):
        self.cols=cols; self.min_count=min_count; self.min_prop=min_prop
    def fit(self, X, y=None):
        n = len(X); thr = max(self.min_count, int(np.ceil(self.min_prop*n)))
        self.keep_maps_ = {}
        for c in self.cols:
            vc = X[c].astype(str).fillna("NA").value_counts()
            self.keep_maps_[c] = set(vc[vc >= thr].index.tolist())
        return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            keep = self.keep_maps_.get(c, set())
            X[c] = X[c].astype(str).fillna("NA").where(X[c].astype(str).isin(keep), "RARE")
        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop): self.cols_to_drop=cols_to_drop
    def fit(self, X, y=None): return self
    def transform(self, X): return X.drop(columns=self.cols_to_drop, errors="ignore")

def build_preprocessor():
    num_cols = ["Age","Fare","SibSp","Parch","FamilySize","FarePerPerson","IsAlone"]
    cat_cols = ["Sex","Embarked","Pclass","Title","Deck","Pclass_Sex"]
    prep = Pipeline([
        ("fe", FeatureEngineer()),
        ("rare", RareCategoryGrouper(cols=["Title","Deck","Embarked"], min_count=8, min_prop=0.01)),
        ("drop", ColumnDropper(cols_to_drop=[ID_COL,"Name","Ticket","Ticket_norm","Cabin"])),
        ("ct", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))]), cat_cols)
        ], remainder="drop", sparse_threshold=0.0))
    ])
    return prep
