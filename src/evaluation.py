"""
evaluation.py
Student Dropout Prediction System
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Evaluate a trained model on the test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'model': model_name,
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'f1_score': round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, y_prob) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k not in ['model', 'confusion_matrix']:
            print(f"  {k.upper():<15}: {v:.2f}%")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

    return metrics

def compare_models(models_dict: dict, X_test, y_test) -> list:
    """Compare multiple models and return sorted metrics."""
    all_metrics = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)

    all_metrics.sort(key=lambda x: x['accuracy'], reverse=True)
    print("\n[INFO] Model Ranking by Accuracy:")
    for i, m in enumerate(all_metrics, 1):
        print(f"  {i}. {m['model']}: {m['accuracy']}%")

    return all_metrics
