"""
main.py
Student Dropout Prediction System
Entry point: trains models, evaluates, and launches app.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import engineer_features
from src.model_training import train_random_forest, train_xgboost
from src.evaluation import compare_models

import pandas as pd

def train_and_evaluate():
    print("\n" + "="*60)
    print("  STUDENT DROPOUT PREDICTION SYSTEM")
    print("  Training Pipeline")
    print("="*60)

    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        raw_path='data/raw/student_uci.csv',
        processed_path='data/processed/final_dataset.csv'
    )

    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    results = compare_models(
        {'Random Forest': rf_model, 'XGBoost': xgb_model},
        X_test, y_test
    )

    print("\n[INFO] Training complete. Starting web application...")
    return results

def run_app():
    os.chdir(os.path.dirname(__file__))
    train_and_evaluate()
    from app.app import app
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    if "--train-only" in sys.argv:
        train_and_evaluate()
    else:
        run_app()
