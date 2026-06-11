"""
feature_engineering.py
Student Dropout Prediction System
"""

import pandas as pd
import numpy as np

def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite risk score from key features."""
    df = df.copy()
    
    # Normalize and invert risk indicators (higher = more risk)
    df['gpa_risk'] = 1 - (df['gpa'] / 10.0)                          # GPA 0-10
    df['attendance_risk'] = 1 - (df['attendance_rate'] / 100.0)       # Attendance 0-100%
    df['income_risk'] = 1 - (df['family_income'] / df['family_income'].max())
    df['parental_edu_risk'] = 1 - (df['parent_education'] / df['parent_education'].max())

    df['composite_risk_score'] = (
        0.35 * df['gpa_risk'] +
        0.30 * df['attendance_risk'] +
        0.20 * df['income_risk'] +
        0.15 * df['parental_edu_risk']
    ).round(4)

    print("[INFO] Composite risk score computed.")
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction terms between features."""
    df = df.copy()
    df['gpa_x_attendance'] = (df['gpa'] / 10.0) * (df['attendance_rate'] / 100.0)
    df['income_x_edu'] = df['family_income'] * df['parent_education']
    df['failure_x_distance'] = df['previous_failures'] * df['school_distance_km']
    print("[INFO] Interaction features added.")
    return df

def add_categorical_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary risk flag columns based on thresholds."""
    df = df.copy()
    df['low_gpa_flag'] = (df['gpa'] < 4.0).astype(int)
    df['low_attendance_flag'] = (df['attendance_rate'] < 40.0).astype(int)
    df['low_income_flag'] = (df['family_income'] < 15000).astype(int)
    df['high_failure_flag'] = (df['previous_failures'] >= 2).astype(int)
    df['no_internet_flag'] = (df['internet_access'] == 0).astype(int)
    print("[INFO] Categorical risk flags added.")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master feature engineering function."""
    df = add_risk_score(df)
    df = add_interaction_features(df)
    df = add_categorical_flags(df)
    print(f"[INFO] Final feature count: {df.shape[1]}")
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/processed/final_dataset.csv')
    df_engineered = engineer_features(df)
    df_engineered.to_csv('data/processed/final_dataset.csv', index=False)
    print("[INFO] Feature-engineered dataset saved.")
