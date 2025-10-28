
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Define the preprocessing pipeline
numerical_features = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'GPA']
categorical_features = ['University Rating', 'Research']

# Numerical pipeline
def create_numerical_pipeline():
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

# Categorical pipeline
def create_categorical_pipeline():
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

# Column Transformer to combine both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', create_numerical_pipeline(), numerical_features),
        ('cat', create_categorical_pipeline(), categorical_features)
    ]
)

# Full Pipeline with SMOTE
full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE()),
    ('classifier', RandomForestClassifier())
])

# Save pipeline to file
import joblib
joblib.dump(full_pipeline, 'models/transformers.joblib')

# Function to fit and transform training data
if __name__ == "__main__":
    train_data = pd.read_parquet('data/processed/train.parquet')
    X_train = train_data.drop('Chance of Admit', axis=1)
    y_train = train_data['Chance of Admit']
    full_pipeline.fit(X_train, y_train)
