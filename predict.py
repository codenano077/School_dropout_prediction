"""
predict.py
Student Dropout Prediction System
"""

import pickle
import numpy as np
import pandas as pd

def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_scaler(scaler_path: str = 'models/scaler.pkl'):
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

FEATURE_ORDER = [
    'age', 'gender', 'gpa', 'attendance_rate', 'family_income',
    'parent_education', 'study_hours', 'extracurricular',
    'previous_failures', 'school_distance_km', 'siblings',
    'internet_access', 'single_parent'
]

def predict_single(student_data: dict, model_path: str, scaler_path: str = 'models/scaler.pkl'):
    """Predict dropout probability for a single student."""
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    row = [student_data.get(f, 0) for f in FEATURE_ORDER]
    X = np.array(row).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    result = {
        'dropout_prediction': int(prediction),
        'dropout_probability': round(float(probability[1]) * 100, 2),
        'safe_probability': round(float(probability[0]) * 100, 2),
        'risk_level': get_risk_level(float(probability[1]))
    }
    return result

def get_risk_level(prob: float) -> str:
    if prob >= 0.75:
        return 'CRITICAL'
    elif prob >= 0.55:
        return 'HIGH'
    elif prob >= 0.35:
        return 'MODERATE'
    else:
        return 'LOW'

def predict_batch(df: pd.DataFrame, model_path: str, scaler_path: str = 'models/scaler.pkl'):
    """Predict dropout for a batch of students."""
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    X = df[FEATURE_ORDER].values
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    df = df.copy()
    df['dropout_prediction'] = predictions
    df['dropout_probability'] = (probabilities * 100).round(2)
    df['risk_level'] = [get_risk_level(p) for p in probabilities]
    return df
