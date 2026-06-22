import numpy as np
import pandas as pd
import json
import os
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from src.leakage_detector import DataLeakageDetector

def validate_parkinsons_regression():
    print("--- Regression Test: Parkinson's Dataset ---")
    data_path = "data/raw/dataset.csv"
    if not os.path.exists(data_path):
        print(f"Skipping Parkinson's regression test: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    # Target column is likely 'Class' or 'Status', need to confirm. Assuming 'Class' based on typical PD datasets
    target_col = 'PC'
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in PD dataset.")
        return
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    detector = DataLeakageDetector()
    detector.fit(X, y)
    scores = detector.get_scores()
    
    print("Top 5 Suspicious Features in PD Dataset:")
    print(scores.head())
    
    # Assert Duration is caught by degeneracy
    if 'Duration' in scores['feature'].values:
        dur_stats = scores[scores['feature'] == 'Duration'].iloc[0]
        if dur_stats['degenerate_flag']:
            print("[PASS] 'Duration' successfully caught via degeneracy flag.")
        else:
            print("[FAIL] 'Duration' found but NOT flagged as degenerate.")
    else:
        print("[FAIL] 'Duration' feature not found in dataset.")


def validate_breast_cancer():
    print("\n--- Validation: Breast Cancer Wisconsin ---")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    detector = DataLeakageDetector()
    detector.fit(X, y)
    clean_scores = detector.get_scores()
    
    print("Top 5 Legitimate Strong Features (Should NOT be degenerate):")
    print(clean_scores.head())
    
    # Check false positives on degeneracy
    if clean_scores['degenerate_flag'].sum() > 0:
        print("[FAIL] Found degenerate flags in clean Breast Cancer dataset.")
    else:
        print("[PASS] No false positive degeneracy flags on legitimate strong features.")
        
    # Inject subtle leak (noisy proxy of target)
    np.random.seed(42)
    # A feature that has ~0.7-0.8 correlation with target
    subtle_leak = y + np.random.normal(0, 0.5, size=len(y))
    X_leaked = X.copy()
    X_leaked['INJECTED_PROXY_LEAK'] = subtle_leak
    
    # Inject degeneracy leak
    deg_leak = np.zeros(len(y))
    deg_leak[y == 1] = np.random.normal(5, 1, size=(y == 1).sum())
    X_leaked['INJECTED_DEG_LEAK'] = deg_leak
    
    detector_leaked = DataLeakageDetector()
    detector_leaked.fit(X_leaked, y)
    leaked_scores = detector_leaked.get_scores()
    
    print("\nTop 5 Features after Injection:")
    print(leaked_scores.head())
    
    # Ground truth for PR curve (Is it a leak?)
    # We define only the injected leaks as True positives
    y_true_leaks = leaked_scores['feature'].str.startswith('INJECTED').astype(int)
    y_scores_pred = leaked_scores['predictiveness_score']
    
    precision, recall, thresholds = precision_recall_curve(y_true_leaks, y_scores_pred)
    pr_auc = auc(recall, precision)
    print(f"\nPR AUC for PR curve across thresholds: {pr_auc:.3f}")
    
    # The proxy leak should NOT trigger degeneracy, but deg_leak MUST trigger degeneracy
    proxy_row = leaked_scores[leaked_scores['feature'] == 'INJECTED_PROXY_LEAK'].iloc[0]
    deg_row = leaked_scores[leaked_scores['feature'] == 'INJECTED_DEG_LEAK'].iloc[0]
    
    if proxy_row['degenerate_flag']:
        print("[FAIL] Proxy leak incorrectly flagged as degenerate.")
    else:
        print("[PASS] Proxy leak NOT flagged as degenerate (honest scoping).")
        
    if deg_row['degenerate_flag']:
        print("[PASS] Degeneracy leak successfully caught.")
    else:
        print("[FAIL] Degeneracy leak missed.")


def validate_wine_quality():
    print("\n--- Validation: Wine Quality (Binarized) ---")
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    # Binarize: Class 0 vs Class 1 & 2
    y = (data.target == 0).astype(int)
    
    # Inject subtle leak
    np.random.seed(42)
    subtle_leak = y + np.random.normal(0, 0.8, size=len(y)) # Weak univariate signal
    X_leaked = X.copy()
    X_leaked['INJECTED_WEAK_PROXY_LEAK'] = subtle_leak
    
    detector = DataLeakageDetector()
    detector.fit(X_leaked, y)
    scores = detector.get_scores()
    
    print("Top 5 Features after Injection:")
    print(scores.head())
    
    y_true_leaks = scores['feature'].str.startswith('INJECTED').astype(int)
    y_scores_pred = scores['predictiveness_score']
    
    precision, recall, thresholds = precision_recall_curve(y_true_leaks, y_scores_pred)
    pr_auc = auc(recall, precision)
    print(f"PR AUC for PR curve across thresholds: {pr_auc:.3f}")


if __name__ == "__main__":
    validate_parkinsons_regression()
    validate_breast_cancer()
    validate_wine_quality()
