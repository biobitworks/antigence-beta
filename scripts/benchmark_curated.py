#!/usr/bin/env python3
"""
Antigence Benchmark — Curated Security Patterns

Trains and tests NegSl-AIS, B-Cell, NK Cell, and GuardrailPipeline against
the curated safe/vulnerable code patterns (60 samples, 8 CWE types).

This is a more realistic benchmark than synthetic single-line samples.

Usage:
    PYTHONPATH=src python3 scripts/benchmark_curated.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "code_security_scanner"))

from datasets.safe_patterns import SAFE_PATTERNS
from datasets.vulnerable_patterns import VULNERABLE_PATTERNS

from immunos_mcp.algorithms.negsel import (
    NegativeSelectionClassifier,
    NegSelConfig,
    calculate_mcc,
    calculate_kappa,
)
from immunos_mcp.agents.bcell_agent import BCellAgent
from immunos_mcp.agents.nk_cell_agent import NKCellAgent
from immunos_mcp.agents.dendritic_agent import DendriticAgent
from immunos_mcp.core.antigen import Antigen
from immunos_mcp.guardrails import GuardrailPipeline, GuardrailConfig


def extract_code(patterns):
    """Extract code strings from pattern dicts."""
    return [p["code"] for p in patterns]


def make_antigens(code_list, label):
    """Create Antigen objects from code strings."""
    return [Antigen.from_code(c, class_label=label) for c in code_list]


def split_train_test(items, train_ratio=0.6):
    """Split list into train/test sets."""
    n = int(len(items) * train_ratio)
    return items[:n], items[n:]


def print_metrics(name, y_true, y_pred, elapsed):
    """Print standard classification metrics."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    mcc = calculate_mcc(tp, tn, fp, fn)
    kappa = calculate_kappa(tp, tn, fp, fn)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Samples:   {len(y_true)} (safe={tn+fp}, vuln={tp+fn})")
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
        "samples": len(y_true),
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
    """Benchmark NegSl-AIS on curated patterns."""
    print("\n--- Training NegSl-AIS on safe patterns ---")
    t0 = time.time()

    safe_antigens = make_antigens(safe_train, "safe")
    features = [dendritic.get_feature_vector(a) for a in safe_antigens]

    clf = NegativeSelectionClassifier(config=NegSelConfig(
        r_self=0.3,
        num_detectors=200,
    ))
    clf.fit(np.array(features))

    # Test
    test_safe_antigens = make_antigens(safe_test, "safe")
    test_vuln_antigens = make_antigens(vuln_test, "vulnerable")

    y_true = []
    y_pred = []

    for a in test_safe_antigens:
        fv = dendritic.get_feature_vector(a)
        result = clf.predict_single(fv)
        y_true.append(0)
        y_pred.append(1 if result else 0)

    for a in test_vuln_antigens:
        fv = dendritic.get_feature_vector(a)
        result = clf.predict_single(fv)
        y_true.append(1)
        y_pred.append(1 if result else 0)

    elapsed = time.time() - t0
    return print_metrics("NegSl-AIS (Curated Patterns)", y_true, y_pred, elapsed)


def benchmark_bcell(safe_train, vuln_train, safe_test, vuln_test):
    """Benchmark B-Cell classifier on curated patterns."""
    print("\n--- Training B-Cell on labeled patterns ---")
    t0 = time.time()

    safe_antigens = make_antigens(safe_train, "safe")
    vuln_antigens = make_antigens(vuln_train, "unsafe")

    bcell = BCellAgent(agent_name="bench_bcell", affinity_method="traditional")
    bcell.train(safe_antigens + vuln_antigens)

    # Test
    y_true = []
    y_pred = []

    for code in safe_test:
        a = Antigen.from_code(code)
        result = bcell.recognize(a)
        y_true.append(0)
        y_pred.append(1 if result.predicted_class == "unsafe" else 0)

    for code in vuln_test:
        a = Antigen.from_code(code)
        result = bcell.recognize(a)
        y_true.append(1)
        y_pred.append(1 if result.predicted_class == "unsafe" else 0)

    elapsed = time.time() - t0
    return print_metrics("B-Cell Classifier (Curated Patterns)", y_true, y_pred, elapsed)


def benchmark_guardrail(safe_train, vuln_train, safe_test, vuln_test):
    """Benchmark GuardrailPipeline on curated patterns."""
    print("\n--- Training GuardrailPipeline ---")
    t0 = time.time()

    pipeline = GuardrailPipeline(config=GuardrailConfig(
        block_on_anomaly=True,
        block_on_danger=True,
    ))
    pipeline.train_on_safe_examples(safe_train, is_code=True)
    pipeline.train_classifier(safe_train, vuln_train, is_code=True)

    # Test
    y_true = []
    y_pred = []

    for code in safe_test:
        result = pipeline.validate_code(code)
        y_true.append(0)
        y_pred.append(1 if result.blocked else 0)

    for code in vuln_test:
        result = pipeline.validate_code(code)
        y_true.append(1)
        y_pred.append(1 if result.blocked else 0)

    elapsed = time.time() - t0
    return print_metrics("GuardrailPipeline (Full Stack)", y_true, y_pred, elapsed)


def main():
    print("=" * 60)
    print("  Antigence Benchmark — Curated Security Patterns")
    print("=" * 60)

    safe_code = extract_code(SAFE_PATTERNS)
    vuln_code = extract_code(VULNERABLE_PATTERNS)

    print(f"\nDataset: {len(safe_code)} safe, {len(vuln_code)} vulnerable")
    print(f"CWE types: {len(set(p.get('cwe_id', 'N/A') for p in VULNERABLE_PATTERNS))}")

    # Split 60/40
    safe_train, safe_test = split_train_test(safe_code)
    vuln_train, vuln_test = split_train_test(vuln_code)
    print(f"Train: {len(safe_train)} safe, {len(vuln_train)} vuln")
    print(f"Test:  {len(safe_test)} safe, {len(vuln_test)} vuln")

    dendritic = DendriticAgent(agent_name="bench_dendritic")

    results = []
    results.append(benchmark_negsel(safe_train, vuln_test, safe_test, dendritic))
    results.append(benchmark_bcell(safe_train, vuln_train, safe_test, vuln_test))
    results.append(benchmark_guardrail(safe_train, vuln_train, safe_test, vuln_test))

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY — Curated Security Patterns")
    print(f"{'='*60}")
    print(f"{'Algorithm':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6}")
    print("-" * 71)
    for r in results:
        print(f"{r['name']:<35} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
              f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['fpr']:>6.3f}")

    # Save results
    out_dir = Path(__file__).resolve().parent.parent
    with open(out_dir / "BENCHMARKS_CURATED.json", "w") as f:
        json.dump({"benchmark": "curated_security_patterns", "results": results}, f, indent=2)

    print(f"\nResults saved to BENCHMARKS_CURATED.json")


if __name__ == "__main__":
    main()
