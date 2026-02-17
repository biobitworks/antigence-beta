# Antigence Benchmark Results

**Date**: 2026-02-16T17:19:18Z
**Dataset**: 20 safe + 20 vulnerable Python code snippets
**Seed**: 42 (reproducible)

## Results

| Algorithm | Accuracy | Precision | Recall | F1 | MCC | FPR |
|-----------|----------|-----------|--------|----|-----|-----|
| NegSl-AIS | 0.6750 | 1.0000 | 0.3500 | 0.5185 | 0.4606 | 0.0000 |
| B-Cell (SHA) | 0.5833 | 0.5455 | 1.0000 | 0.7059 | 0.3015 | N/A |
| NK Cell (Feature) | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Key Properties

- **NegSl-AIS False Positive Rate**: 0.0000 — Self-data is never misclassified (deterministic guarantee)
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
