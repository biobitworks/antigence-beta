#!/usr/bin/env python3
"""
Antigence Benchmark — Devign Dataset (27,318 real C/C++ functions)

Trains and tests NegSl-AIS, B-Cell, and GuardrailPipeline against
real-world labeled vulnerability data from FFmpeg and QEMU.

Source: https://github.com/epicosy/devign
Labels: 0 = safe, 1 = vulnerable

Usage:
    PYTHONPATH=src python3 scripts/benchmark_devign.py
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from immunos_mcp.algorithms.negsel import (
    NegativeSelectionClassifier,
    NegSelConfig,
    calculate_mcc,
    calculate_kappa,
)
from immunos_mcp.agents.bcell_agent import BCellAgent
from immunos_mcp.agents.dendritic_agent import DendriticAgent
from immunos_mcp.core.antigen import Antigen
from immunos_mcp.guardrails import GuardrailPipeline, GuardrailConfig


DEVIGN_PATH = Path(__file__).resolve().parent.parent / "data" / "datasets" / "devign" / "data" / "raw" / "dataset.json"

# How many samples to use (full dataset is 27K — subsample for speed)
TRAIN_SAFE = 200
TRAIN_VULN = 200
TEST_SAFE = 100
TEST_VULN = 100

SEED = 42


def load_devign():
    """Load Devign dataset and split into safe/vulnerable."""
    print(f"Loading Devign from {DEVIGN_PATH}...")
    with open(DEVIGN_PATH) as f:
        data = json.load(f)

    safe = [d["func"] for d in data if d["target"] == 0]
    vuln = [d["func"] for d in data if d["target"] == 1]

    print(f"  Total: {len(data)} ({len(safe)} safe, {len(vuln)} vulnerable)")
    return safe, vuln


def subsample(safe, vuln):
    """Subsample for benchmark speed."""
    random.seed(SEED)
    random.shuffle(safe)
    random.shuffle(vuln)

    safe_train = safe[:TRAIN_SAFE]
    safe_test = safe[TRAIN_SAFE:TRAIN_SAFE + TEST_SAFE]
    vuln_train = vuln[:TRAIN_VULN]
    vuln_test = vuln[TRAIN_VULN:TRAIN_VULN + TEST_VULN]

    print(f"  Subsampled: train={TRAIN_SAFE}+{TRAIN_VULN}, test={TEST_SAFE}+{TEST_VULN}")
    return safe_train, safe_test, vuln_train, vuln_test


def print_metrics(name, y_true, y_pred, elapsed):
    """Print classification metrics."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    n = len(y_true)
    accuracy = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    mcc = calculate_mcc(tp, tn, fp, fn)
    kappa = calculate_kappa(tp, tn, fp, fn)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Samples:   {n} (safe={tn+fp}, vuln={tp+fn})")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  FPR:       {fpr:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    print(f"  Kappa:     {kappa:.4f}")
    print(f"  Time:      {elapsed:.2f}s")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    return {
        "name": name,
        "dataset": "devign",
        "samples": n,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "mcc": round(mcc, 4),
        "kappa": round(kappa, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "elapsed_s": round(elapsed, 2),
    }


def benchmark_negsel(safe_train, vuln_test, safe_test, dendritic):
    """Benchmark NegSl-AIS."""
    print("\n--- NegSl-AIS: Training on safe C/C++ functions ---")
    t0 = time.time()

    # Extract features from training safe samples
    safe_antigens = [Antigen.from_code(c, language="c", class_label="safe") for c in safe_train]
    features = np.array([dendritic.get_feature_vector(a) for a in safe_antigens])

    clf = NegativeSelectionClassifier(config=NegSelConfig(
        r_self=0.3,
        num_detectors=200,
    ))
    clf.fit(features)
    print(f"  Generated {len(clf.valid_detectors)} detectors from {len(safe_train)} safe functions")

    # Test
    y_true, y_pred = [], []

    for code in safe_test:
        a = Antigen.from_code(code, language="c")
        fv = dendritic.get_feature_vector(a)
        score = clf.predict_single(fv)
        y_true.append(0)
        y_pred.append(1 if score > 0 else 0)

    for code in vuln_test:
        a = Antigen.from_code(code, language="c")
        fv = dendritic.get_feature_vector(a)
        score = clf.predict_single(fv)
        y_true.append(1)
        y_pred.append(1 if score > 0 else 0)

    elapsed = time.time() - t0
    return print_metrics("NegSl-AIS (Devign)", y_true, y_pred, elapsed)


def benchmark_bcell(safe_train, vuln_train, safe_test, vuln_test):
    """Benchmark B-Cell classifier."""
    print("\n--- B-Cell: Training on labeled C/C++ functions ---")
    t0 = time.time()

    safe_antigens = [Antigen.from_code(c, language="c", class_label="safe") for c in safe_train]
    vuln_antigens = [Antigen.from_code(c, language="c", class_label="unsafe") for c in vuln_train]

    bcell = BCellAgent(agent_name="devign_bcell", affinity_method="traditional")
    bcell.train(safe_antigens + vuln_antigens)

    # Test
    y_true, y_pred = [], []

    for code in safe_test:
        a = Antigen.from_code(code, language="c")
        result = bcell.recognize(a)
        y_true.append(0)
        y_pred.append(1 if result.predicted_class == "unsafe" else 0)

    for code in vuln_test:
        a = Antigen.from_code(code, language="c")
        result = bcell.recognize(a)
        y_true.append(1)
        y_pred.append(1 if result.predicted_class == "unsafe" else 0)

    elapsed = time.time() - t0
    return print_metrics("B-Cell (Devign)", y_true, y_pred, elapsed)


def benchmark_guardrail(safe_train, vuln_train, safe_test, vuln_test):
    """Benchmark GuardrailPipeline."""
    print("\n--- GuardrailPipeline: Full stack on C/C++ functions ---")
    t0 = time.time()

    pipeline = GuardrailPipeline(config=GuardrailConfig(
        block_on_anomaly=True,
        block_on_danger=True,
    ))
    pipeline.train_on_safe_examples(safe_train, is_code=True)
    pipeline.train_classifier(safe_train, vuln_train, is_code=True)

    # Test
    y_true, y_pred = [], []

    for code in safe_test:
        result = pipeline.validate_code(code, language="c")
        y_true.append(0)
        y_pred.append(1 if result.blocked else 0)

    for code in vuln_test:
        result = pipeline.validate_code(code, language="c")
        y_true.append(1)
        y_pred.append(1 if result.blocked else 0)

    elapsed = time.time() - t0
    return print_metrics("GuardrailPipeline (Devign)", y_true, y_pred, elapsed)


def main():
    print("=" * 60)
    print("  Antigence Benchmark — Devign Dataset")
    print("  27,318 real C/C++ functions (FFmpeg, QEMU)")
    print("=" * 60)

    if not DEVIGN_PATH.exists():
        print(f"\nERROR: Devign dataset not found at {DEVIGN_PATH}")
        print("Run: git clone --depth=1 https://github.com/epicosy/devign.git data/datasets/devign")
        sys.exit(1)

    safe, vuln = load_devign()
    safe_train, safe_test, vuln_train, vuln_test = subsample(safe, vuln)

    dendritic = DendriticAgent(agent_name="devign_dendritic")

    results = []
    results.append(benchmark_negsel(safe_train, vuln_test, safe_test, dendritic))
    results.append(benchmark_bcell(safe_train, vuln_train, safe_test, vuln_test))
    results.append(benchmark_guardrail(safe_train, vuln_train, safe_test, vuln_test))

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY — Devign (Real C/C++ Functions)")
    print(f"{'='*60}")
    print(f"  Train: {TRAIN_SAFE} safe + {TRAIN_VULN} vuln")
    print(f"  Test:  {TEST_SAFE} safe + {TEST_VULN} vuln")
    print(f"{'Algorithm':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6}")
    print("-" * 71)
    for r in results:
        print(f"{r['name']:<35} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
              f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['fpr']:>6.3f}")

    # Save
    out_dir = Path(__file__).resolve().parent.parent
    out = {"benchmark": "devign_real_functions", "seed": SEED, "results": results,
           "train_safe": TRAIN_SAFE, "train_vuln": TRAIN_VULN,
           "test_safe": TEST_SAFE, "test_vuln": TEST_VULN}
    with open(out_dir / "BENCHMARKS_DEVIGN.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nResults saved to BENCHMARKS_DEVIGN.json")


if __name__ == "__main__":
    main()
