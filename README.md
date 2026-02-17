# Antigence

**Deterministic guardrails for probabilistic models.**

Antigence is an open-source framework that applies Artificial Immune System (AIS) algorithms to validate AI model outputs. It provides mathematical guarantees that probabilistic systems cannot — including zero false positives on known-safe data, cryptographic integrity of agent state, and deterministic feature extraction.

[![CI](https://github.com/biobitworks/antigence-beta/actions/workflows/ci.yml/badge.svg)](https://github.com/biobitworks/antigence-beta/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## What It Does

```python
from immunos_mcp.guardrails import GuardrailPipeline

pipeline = GuardrailPipeline()
result = pipeline.validate_output("This miracle cure guarantees 100% success!")

print(result.blocked)      # True
print(result.risk_level)   # "HIGH"
print(result.danger_score) # 0.67
print(result.reason)       # "danger_signals(0.67)"
```

Train on your own data for anomaly detection:

```python
pipeline.train_on_safe_examples([
    "x = 1 + 2",
    "print('hello')",
    "result = sorted(items)",
], is_code=True)

result = pipeline.validate_code("eval(input())")
print(result.anomaly_detected)  # True
```

## Guarantees

See [GUARANTEES.md](GUARANTEES.md) for the full details. In summary:

| Guarantee | Mechanism | Test Coverage |
|-----------|-----------|---------------|
| Self-data never misclassified | NegSl-AIS: R_q > R_self constraint | `test_detectors_never_fire_on_self` |
| Tampered state always rejected | Ed25519 + HMAC-SHA256 signatures | `test_tampered_data_rejected` |
| Same input = same features | Rule-based regex, no randomness | `test_same_input_same_vector_100_runs` |

**Not guaranteed**: Detection of all anomalies (35% recall on synthetic data — see [BENCHMARKS.md](BENCHMARKS.md)).

## Architecture

Bio-inspired multi-agent system where each component mirrors an immune cell:

| Component | Role | What It Does |
|-----------|------|-------------|
| **Dendritic Agent** | Feature Extraction | Deterministic 20-dim vector from text/code |
| **NK Cell Agent** | Anomaly Detection | Negative selection — flags non-self inputs |
| **B Cell Agent** | Pattern Classification | Learns safe vs unsafe patterns |
| **Memory Agent** | Knowledge Cache | Stores and retrieves learned patterns |
| **GuardrailPipeline** | Integration | Chains all agents into a validate/block decision |

```
Input (text/code)
    |
    v
[Dendritic Agent] --> 20-dim feature vector
    |
    v
[Danger Signal Check] --> block if PAMP score > threshold
    |
    v
[NK Cell] --> block if anomaly detected (trained)
    |
    v
[B Cell] --> classify safe/unsafe (trained)
    |
    v
GuardrailResult { blocked, risk_level, scores }
```

## Installation

```bash
pip install -e ".[dev]"

# With GPU acceleration (optional)
pip install -e ".[dev,gpu]"
```

## Quick Start

### Validate LLM Text Output

```python
from immunos_mcp.guardrails import GuardrailPipeline, GuardrailConfig

# Default config blocks on danger signals
pipeline = GuardrailPipeline()

result = pipeline.validate_output(
    "The data suggests further research is needed (DOI: 10.1234/test)."
)
assert result.passed  # Safe — hedged language, citation present

result = pipeline.validate_output(
    "This cures everything guaranteed with no side effects!"
)
assert result.blocked  # Blocked — danger signals detected
```

### Validate Code

```python
result = pipeline.validate_code("x = sorted(items)")
assert result.passed

result = pipeline.validate_code("eval(user_input)")
# May be blocked depending on training and config
```

### Train for Anomaly Detection

```python
pipeline = GuardrailPipeline(config=GuardrailConfig(
    block_on_anomaly=True,
))

# Train on known-safe code
pipeline.train_on_safe_examples([
    "x = 1 + 2",
    "print('hello')",
    "def add(a, b): return a + b",
    "result = sorted(items)",
    "data = json.loads(text)",
], is_code=True)

# Now anomalous code is flagged
result = pipeline.validate_code("os.system(user_cmd)")
print(result.anomaly_detected)
```

### Train a Classifier

```python
pipeline = GuardrailPipeline()

pipeline.train_classifier(
    safe_texts=["x = 1", "print('hello')", "sorted(items)"],
    unsafe_texts=["eval(input())", "os.system(cmd)", "exec(payload)"],
    is_code=True,
)

result = pipeline.validate_code("pickle.loads(data)")
print(result.classification)            # "safe" or "unsafe"
print(result.classification_confidence) # 0.0 - 1.0
```

## Benchmarks

Current results on synthetic code security data (see [BENCHMARKS.md](BENCHMARKS.md)):

| Algorithm | Accuracy | Precision | FPR | Recall |
|-----------|----------|-----------|-----|--------|
| NegSl-AIS | 0.675 | 1.000 | **0.000** | 0.350 |
| B Cell | 0.583 | 0.600 | 0.350 | 0.600 |
| NK Cell | 0.500 | 1.000 | **0.000** | — |

Key takeaway: **Zero false positive rate is guaranteed by construction** (NegSl-AIS). Recall improves with better training data and embedding-based features.

## Testing

```bash
# Run all tests (100 tests)
pytest tests/ -v --ignore=tests/smoke_test.py

# With coverage
pytest tests/ --cov=immunos_mcp --cov-report=term-missing --ignore=tests/smoke_test.py
```

## Project Structure

```
antigence/
├── src/immunos_mcp/
│   ├── guardrails/          # GuardrailPipeline (start here)
│   ├── agents/              # NK Cell, B Cell, Dendritic, Memory
│   ├── algorithms/          # NegSl-AIS, Opt-AiNet
│   ├── core/                # Antigen, Affinity, Protocols
│   └── security/            # Ed25519 signer/verifier
├── tests/                   # 100 falsification tests
├── scripts/                 # Training and benchmark scripts
├── web_app/                 # Flask dashboard
├── GUARANTEES.md            # What we guarantee (and don't)
└── BENCHMARKS.md            # Published accuracy results
```

## Publications

- Antigence: Hypothesis and Software Overview (Zenodo, 2025) — DOI: 10.5281/zenodo.18109862

## Mathematical Foundation

- **NegSl-AIS**: Umair et al. (2025), "Negative selection-based artificial immune system", *Results in Engineering* 27, 106601
- **Opt-AiNet**: de Castro & Timmis (2002), immune network optimization
- **Affinity**: Immunos-81 formula (Hunt & Cooke, 2000) with exponential decay and cosine similarity
- **Dendritic Cell Algorithm**: Greensmith et al. (2005)

## License

Apache 2.0
