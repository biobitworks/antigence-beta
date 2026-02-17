# Antigence Benchmark Results

## Benchmark 1: Synthetic Code Snippets

**Date**: 2026-02-16
**Dataset**: 20 safe + 20 vulnerable single-line Python code snippets
**Features**: Dendritic 20-dim feature vectors (regex-based)

| Algorithm | Accuracy | Precision | Recall | F1 | FPR |
|-----------|----------|-----------|--------|----|-----|
| NegSl-AIS | 0.675 | 1.000 | 0.350 | 0.519 | **0.000** |
| B-Cell (SHA) | 0.583 | 0.545 | 1.000 | 0.706 | N/A |
| NK Cell (Feature) | 0.500 | — | — | — | **0.000** |

## Benchmark 2: Curated Security Patterns

**Date**: 2026-02-16
**Dataset**: 30 safe + 30 vulnerable multi-line Python code (real-world patterns)
**CWE types**: CWE-89, CWE-78, CWE-22, CWE-79, CWE-798, CWE-502, CWE-327, CWE-918
**Split**: 60% train / 40% test

| Algorithm | Accuracy | Precision | Recall | F1 | FPR |
|-----------|----------|-----------|--------|----|-----|
| NegSl-AIS | 0.333 | 0.250 | 0.167 | 0.200 | 0.500 |
| B-Cell | **0.750** | **1.000** | 0.500 | 0.667 | **0.000** |
| GuardrailPipeline | **0.750** | **1.000** | 0.500 | 0.667 | **0.000** |

## Benchmark 3: Devign — Real C/C++ Functions

**Date**: 2026-02-16
**Dataset**: 27,318 real C/C++ functions from FFmpeg and QEMU (Devign, 2019)
**Subsample**: 200 safe + 200 vuln train, 100 safe + 100 vuln test
**Features**: Dendritic 20-dim feature vectors (regex-based)

| Algorithm | Accuracy | Precision | Recall | F1 | FPR |
|-----------|----------|-----------|--------|----|-----|
| NegSl-AIS | 0.495 | 0.496 | 0.650 | 0.563 | 0.660 |
| B-Cell | 0.515 | 0.510 | 0.790 | 0.620 | 0.760 |
| GuardrailPipeline | 0.505 | 0.503 | 0.790 | 0.615 | 0.780 |

**Result: ~50% accuracy (random chance)**. The regex-based Dendritic features cannot detect semantic vulnerabilities (buffer overflows, integer overflows, race conditions) in real C/C++ code. This benchmark establishes the baseline that embedding-based features must beat.

## Where Antigence Works vs Where It Doesn't

| Use Case | Works? | Why |
|----------|--------|-----|
| Known-bad code patterns (`eval()`, `os.system()`, SQL injection) | **Yes** — 75-100% precision | Regex features match these structural patterns directly |
| LLM output validation (danger signals, credibility) | **Yes** — blocks misinformation patterns | Dendritic features designed for text danger signals |
| Semantic C/C++ vulnerability detection | **No** — random chance | Regex features cannot capture buffer overflows, race conditions, etc. |
| Novel vulnerability types not in training data | **Partial** — NegSl-AIS can flag anomalies | Zero-FPR guarantee on training data; detection depends on feature quality |

## What's Needed to Improve

1. **Embedding-based features**: Replace or augment Dendritic regex features with CodeBERT, VulBERTa, or local Ollama embeddings. The AIS algorithms (NegSl-AIS, B-Cell) are mathematically sound — the bottleneck is feature extraction.
2. **Language-specific feature extractors**: Current features were designed for Python. C/C++ needs different structural patterns (pointer arithmetic, memory allocation, bounds checking).
3. **Larger training sets**: The curated benchmark uses only 60 samples. Devign provides 27K for scaling experiments.

## Reproduction

```bash
# Synthetic benchmark
PYTHONPATH=src python3 scripts/benchmark_accuracy.py

# Curated patterns benchmark
PYTHONPATH=src python3 scripts/benchmark_curated.py

# Devign real-world benchmark
PYTHONPATH=src python3 scripts/benchmark_devign.py
```

## Datasets

| Dataset | Samples | Language | Source |
|---------|---------|----------|--------|
| Synthetic | 40 | Python | Hand-written |
| Curated Security Patterns | 60 | Python | 8 CWE types, real-world patterns |
| Devign | 27,318 | C/C++ | FFmpeg + QEMU (github.com/epicosy/devign) |
