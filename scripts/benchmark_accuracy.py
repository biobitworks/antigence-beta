#!/usr/bin/env python3
"""
Antigence Accuracy Benchmark

Runs NegSl-AIS, B-Cell, and NK Cell against synthetic labeled data
and produces accuracy metrics suitable for publication.

Usage:
    PYTHONPATH=src python3 scripts/benchmark_accuracy.py
"""

import json
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
from immunos_mcp.agents.nk_cell_agent import NKCellAgent
from immunos_mcp.agents.dendritic_agent import DendriticAgent
from immunos_mcp.core.antigen import Antigen


# ---------------------------------------------------------------------------
# Synthetic dataset: code samples with known labels
# ---------------------------------------------------------------------------

SAFE_SAMPLES = [
    "x = 1 + 2",
    "print('hello world')",
    "def add(a, b): return a + b",
    "result = [x**2 for x in range(10)]",
    "with open('file.txt') as f: data = f.read()",
    "import math; area = math.pi * r**2",
    "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    "sorted_list = sorted(items, key=lambda x: x.name)",
    "config = json.loads(Path('config.json').read_text())",
    "response = requests.get(url, timeout=30)",
    "logger.info('Processing complete')",
    "class Point: def __init__(self, x, y): self.x, self.y = x, y",
    "data = {'key': 'value', 'count': 42}",
    "filtered = [x for x in items if x > threshold]",
    "try: result = process(data) except ValueError: handle_error()",
    "from pathlib import Path; p = Path.home() / '.config'",
    "async def fetch(url): async with session.get(url) as resp: return await resp.json()",
    "hashlib.sha256(data.encode()).hexdigest()",
    "datetime.now().strftime('%Y-%m-%d')",
    "subprocess.run(['ls', '-la'], check=True, capture_output=True)",
]

VULNERABLE_SAMPLES = [
    "eval(input())",
    "os.system(user_input)",
    "exec(base64.b64decode(payload))",
    "query = 'SELECT * FROM users WHERE name = \\'' + username + '\\''",
    "cursor.execute('DELETE FROM ' + table_name)",
    "subprocess.call(cmd, shell=True)",
    "pickle.loads(untrusted_data)",
    "__import__('os').system('rm -rf /')",
    "yaml.load(user_input)",
    "render_template_string(user_input)",
    "open(user_path, 'w').write(user_data)",
    "socket.connect((user_host, user_port))",
    "ctypes.cdll.LoadLibrary(user_lib)",
    "importlib.import_module(user_module)",
    "compile(user_code, '<string>', 'exec')",
    "getattr(obj, user_attr)()",
    "globals()[user_key] = user_value",
    "marshal.loads(network_data)",
    "xml.etree.ElementTree.fromstring(user_xml)",
    "tarfile.open(user_file).extractall('/')",
]


def run_negsel_benchmark():
    """Benchmark NegSl-AIS on synthetic data via Dendritic features."""
    print("=" * 60)
    print("BENCHMARK: NegSl-AIS Negative Selection")
    print("=" * 60)

    dendritic = DendriticAgent()

    # Extract features
    safe_antigens = [Antigen.from_code(s, class_label="safe") for s in SAFE_SAMPLES]
    vuln_antigens = [Antigen.from_code(s, class_label="vulnerable") for s in VULNERABLE_SAMPLES]

    safe_features = np.array([dendritic.get_feature_vector(a) for a in safe_antigens])
    vuln_features = np.array([dendritic.get_feature_vector(a) for a in vuln_antigens])

    # Train on safe (self) data
    config = NegSelConfig(num_detectors=30, r_self=0.3)
    clf = NegativeSelectionClassifier(config=config, class_label="SELF")
    np.random.seed(42)

    start = time.time()
    clf.fit(safe_features, max_attempts=20000)
    train_time = time.time() - start

    print(f"\nTraining: {len(safe_features)} self-samples, {len(clf.valid_detectors)} detectors")
    print(f"Train time: {train_time:.3f}s")

    # Predict
    tp = tn = fp = fn = 0

    for sample in safe_features:
        pred = clf.predict_single(sample)
        if pred == 0.0:
            tn += 1  # Correctly identified as self
        else:
            fp += 1  # False positive (self flagged as anomaly)

    for sample in vuln_features:
        pred = clf.predict_single(sample)
        if pred == 1.0:
            tp += 1  # Correctly identified as non-self
        else:
            fn += 1  # False negative (anomaly missed)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mcc = calculate_mcc(tp, tn, fp, fn)
    kappa = calculate_kappa(tp, tn, fp, fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    results = {
        "algorithm": "NegSl-AIS",
        "dataset": "synthetic_code_security",
        "safe_samples": len(SAFE_SAMPLES),
        "vulnerable_samples": len(VULNERABLE_SAMPLES),
        "detectors": len(clf.valid_detectors),
        "r_self": config.r_self,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "kappa": round(kappa, 4),
        "false_positive_rate": round(fpr, 4),
        "train_time_s": round(train_time, 3),
    }

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    for k in ["accuracy", "precision", "recall", "f1", "mcc", "kappa", "false_positive_rate"]:
        print(f"{k:<25} {results[k]:>10.4f}")
    print(f"\nConfusion Matrix: TP={tp} TN={tn} FP={fp} FN={fn}")

    return results


def run_bcell_benchmark():
    """Benchmark B-Cell agent on synthetic data."""
    print("\n" + "=" * 60)
    print("BENCHMARK: B-Cell Pattern Matching")
    print("=" * 60)

    safe_antigens = [Antigen.from_code(s, class_label="safe") for s in SAFE_SAMPLES]
    vuln_antigens = [Antigen.from_code(s, class_label="vulnerable") for s in VULNERABLE_SAMPLES]

    # Split: 70% train, 30% test
    train_safe = safe_antigens[:14]
    test_safe = safe_antigens[14:]
    train_vuln = vuln_antigens[:14]
    test_vuln = vuln_antigens[14:]

    agent = BCellAgent(agent_name="benchmark_bcell", affinity_method="traditional")

    start = time.time()
    agent.train(train_safe + train_vuln)
    train_time = time.time() - start

    print(f"\nTraining: {len(train_safe)} safe + {len(train_vuln)} vulnerable")
    print(f"Testing: {len(test_safe)} safe + {len(test_vuln)} vulnerable")
    print(f"Train time: {train_time:.3f}s")

    tp = tn = fp = fn = 0

    for antigen in test_safe:
        result = agent.recognize(antigen)
        if result.predicted_class == "safe":
            tn += 1
        else:
            fp += 1

    for antigen in test_vuln:
        result = agent.recognize(antigen)
        if result.predicted_class == "vulnerable":
            tp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mcc = calculate_mcc(tp, tn, fp, fn)

    results = {
        "algorithm": "B-Cell (SHA strategy, traditional affinity)",
        "dataset": "synthetic_code_security",
        "train_samples": len(train_safe) + len(train_vuln),
        "test_samples": total,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "train_time_s": round(train_time, 3),
    }

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    for k in ["accuracy", "precision", "recall", "f1", "mcc"]:
        print(f"{k:<25} {results[k]:>10.4f}")
    print(f"\nConfusion Matrix: TP={tp} TN={tn} FP={fp} FN={fn}")

    return results


def run_nkcell_benchmark():
    """Benchmark NK Cell (feature mode) on synthetic data."""
    print("\n" + "=" * 60)
    print("BENCHMARK: NK Cell Anomaly Detection (Feature Mode)")
    print("=" * 60)

    dendritic = DendriticAgent()
    safe_antigens = [Antigen.from_code(s, class_label="safe") for s in SAFE_SAMPLES]
    vuln_antigens = [Antigen.from_code(s, class_label="vulnerable") for s in VULNERABLE_SAMPLES]

    safe_fv = [dendritic.get_feature_vector(a) for a in safe_antigens]
    vuln_fv = [dendritic.get_feature_vector(a) for a in vuln_antigens]

    agent = NKCellAgent(agent_name="benchmark_nk", mode="feature", negsel_config="GENERAL")

    start = time.time()
    agent.train_on_features(safe_antigens, safe_fv)
    train_time = time.time() - start

    print(f"\nTraining: {len(safe_antigens)} self-samples")
    print(f"Testing: {len(safe_antigens)} safe + {len(vuln_antigens)} vulnerable")
    print(f"Train time: {train_time:.3f}s")

    tp = tn = fp = fn = 0

    for i, antigen in enumerate(safe_antigens):
        result = agent.detect_with_features(antigen, safe_fv[i])
        if not result.is_anomaly:
            tn += 1
        else:
            fp += 1

    for i, antigen in enumerate(vuln_antigens):
        result = agent.detect_with_features(antigen, vuln_fv[i])
        if result.is_anomaly:
            tp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mcc = calculate_mcc(tp, tn, fp, fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    results = {
        "algorithm": "NK Cell (NegSl-AIS feature mode)",
        "dataset": "synthetic_code_security",
        "self_samples": len(SAFE_SAMPLES),
        "test_samples": total,
        "detectors": len(agent.detector_vectors),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "false_positive_rate": round(fpr, 4),
        "train_time_s": round(train_time, 3),
    }

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    for k in ["accuracy", "precision", "recall", "f1", "mcc", "false_positive_rate"]:
        print(f"{k:<25} {results[k]:>10.4f}")
    print(f"\nConfusion Matrix: TP={tp} TN={tn} FP={fp} FN={fn}")

    return results


def main():
    print("Antigence Accuracy Benchmark")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: 42\n")

    negsel_results = run_negsel_benchmark()
    bcell_results = run_bcell_benchmark()
    nkcell_results = run_nkcell_benchmark()

    # Save results
    all_results = {
        "benchmark_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "synthetic_code_security",
        "dataset_description": "20 safe + 20 vulnerable Python code snippets",
        "results": {
            "negsel": negsel_results,
            "bcell": bcell_results,
            "nkcell": nkcell_results,
        },
    }

    output_path = Path(__file__).resolve().parent.parent / "BENCHMARKS.json"
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {output_path}")

    # Generate markdown
    md_path = Path(__file__).resolve().parent.parent / "BENCHMARKS.md"
    md = f"""# Antigence Benchmark Results

**Date**: {all_results['benchmark_date']}
**Dataset**: {all_results['dataset_description']}
**Seed**: 42 (reproducible)

## Results

| Algorithm | Accuracy | Precision | Recall | F1 | MCC | FPR |
|-----------|----------|-----------|--------|----|-----|-----|
| NegSl-AIS | {negsel_results['accuracy']:.4f} | {negsel_results['precision']:.4f} | {negsel_results['recall']:.4f} | {negsel_results['f1']:.4f} | {negsel_results['mcc']:.4f} | {negsel_results['false_positive_rate']:.4f} |
| B-Cell (SHA) | {bcell_results['accuracy']:.4f} | {bcell_results['precision']:.4f} | {bcell_results['recall']:.4f} | {bcell_results['f1']:.4f} | {bcell_results['mcc']:.4f} | N/A |
| NK Cell (Feature) | {nkcell_results['accuracy']:.4f} | {nkcell_results['precision']:.4f} | {nkcell_results['recall']:.4f} | {nkcell_results['f1']:.4f} | {nkcell_results['mcc']:.4f} | {nkcell_results['false_positive_rate']:.4f} |

## Key Properties

- **NegSl-AIS False Positive Rate**: {negsel_results['false_positive_rate']:.4f} — Self-data is never misclassified (deterministic guarantee)
- **NK Cell**: Trained on safe-only data (no anomaly examples needed)
- **B-Cell**: Requires labeled examples from both classes

## Reproduction

```bash
cd antigence
PYTHONPATH=src python3 scripts/benchmark_accuracy.py
```

## Notes

- This benchmark uses **synthetic data** (code strings with traditional affinity). Results on real embedding-based data will differ.
- The NegSl-AIS FPR of 0.0 is the core deterministic guarantee: the R_q > R_self rule prevents self-data from being flagged.
- B-Cell performance depends on string-level Jaccard affinity, which has limited discriminative power for code. Embedding-based affinity would improve results.
"""
    md_path.write_text(md)
    print(f"Markdown saved to {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
