#!/usr/bin/env python3
"""
Reproduce NegSl-AIS Publication Results
========================================

Script to recreate the benchmarks from Umair et al. (2025) 
using the updated IMMUNOS Negative Selection engine.

This script uses synthetic feature vectors based on the paper's 
description (balanced classes, feature dimensionality) to verify 
engine logic and metric calculations.
"""

import numpy as np
import time
import sys
import os

# Add src to path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from immunos_mcp.algorithms.negsel import (
    NegativeSelectionClassifier,
    NEGSEL_PRESETS,
    calculate_mcc,
    calculate_kappa,
    calculate_gen_error
)

def run_recreation_benchmark(dimension_key: str):
    print(f"\n[BENCHMARK] Recreating dimension: {dimension_key}")
    print("-" * 50)
    
    config = NEGSEL_PRESETS[dimension_key]
    print(f"Config: {config.description}")
    print(f"Target Detectors: {config.num_detectors}")
    print(f"R_self: {config.r_self}")
    
    # 1. Generate Synthetic 'Self' Data (Bio-signal Feature Space)
    # The paper uses various features (PSD, Entropy, etc.)
    # We generate a cluster centered in a specific hyperspace region
    np.random.seed(42)
    feature_dim = 20 # Representative dim
    self_center = np.random.uniform(0.3, 0.7, feature_dim)
    self_data = self_center + np.random.normal(0, 0.05, (1000, feature_dim))
    self_data = np.clip(self_data, 0, 1)
    
    # 2. Train NegSl-AIS
    clf = NegativeSelectionClassifier(config=config, class_label=dimension_key)
    start_time = time.time()
    clf.fit(self_data)
    train_time = time.time() - start_time
    
    print(f"Generated {len(clf.valid_detectors)} detectors in {train_time:.2f}s")
    
    # 3. Validation (Internal Test Set logic)
    # Generate Non-self data (Anomalies)
    non_self_data = np.random.uniform(0, 1, (500, feature_dim))
    # Filter out anything that accidentally falls into self
    distances_to_self = np.min([np.linalg.norm(non_self_data - s, axis=1) for s in self_data], axis=0)
    non_self_data = non_self_data[distances_to_self > config.r_self]
    
    # Metrics tracking
    # Self (Safe) Test
    self_test = self_data[np.random.choice(len(self_data), min(200, len(self_data)))]
    tp = 0 # Non-self detected as non-self
    tn = 0 # Self detected as self
    fp = 0 # Self detected as non-self
    fn = 0 # Non-self detected as self
    
    # Predict on Self
    for s in self_test:
        pred = clf.predict_single(s)
        if pred == 0.0: tn += 1
        else: fp += 1
        
    # Predict on Non-Self
    for ns in non_self_data[:200]:
        pred = clf.predict_single(ns)
        if pred == 1.0: tp += 1
        else: fn += 1
        
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    mcc = calculate_mcc(tp, tn, fp, fn)
    kappa = calculate_kappa(tp, tn, fp, fn)
    
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"MCC: {mcc:.3f}")
    print(f"Cohen's Kappa: {kappa:.3f}")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

if __name__ == "__main__":
    for dim in ["LA", "HA", "LV", "HV"]:
        run_recreation_benchmark(dim)
