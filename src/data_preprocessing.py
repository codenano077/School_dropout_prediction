"""
data_preprocessing.py
Student Dropout Prediction System
Authors reference: Sulak & Koklu (2023), Jovanovic et al. (IEEE), Christle et al.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import pickle

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw student data from CSV."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} records from {filepath}")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using median/mode imputation."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"[INFO] Missing values handled.")
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns using LabelEncoder."""
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[INFO] Encoded categoricals: {cat_cols}")
    return df

def scale_features(df: pd.DataFrame, target_col: str = 'dropout'):
    """Scale numeric features using StandardScaler."""
    feature_cols = [c for c in df.columns if c != target_col and c != 'student_id']
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"[INFO] Features scaled. Scaler saved.")
    return df_scaled, scaler

def split_data(df: pd.DataFrame, target_col: str = 'dropout', test_size: float = 0.2):
    """Split into train/test sets."""
    feature_cols = [c for c in df.columns if c != target_col and c != 'student_id']
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def preprocess_pipeline(raw_path: str, processed_path: str):
    """Full preprocessing pipeline."""
    df = load_raw_data(raw_path)
    df = handle_missing_values(df)
    df = encode_categoricals(df)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"[INFO] Processed data saved to {processed_path}")

    df_scaled, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df_scaled)
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    preprocess_pipeline(
        raw_path='data/raw/student_uci.csv',
        processed_path='data/processed/final_dataset.csv'
    )
