# Antigence: What We Guarantee (and What We Don't)

## What Antigence IS

Antigence is a **deterministic guardrail framework** for probabilistic AI models, built on
Artificial Immune System (AIS) algorithms. It provides mathematical guarantees about
anomaly detection behavior that probabilistic systems cannot.

## Deterministic Guarantees

### 1. Self-Data Is Never Misclassified (NegSl-AIS)

**Guarantee**: Given a training set S of "self" (normal) data, the Negative Selection
algorithm will **never** classify any sample from S as anomalous.

**Mechanism**: The NegSl-AIS algorithm (Umair et al., 2025) generates detectors in
feature space using the validation rule: `R_q > R_self`. Only detectors that are
sufficiently far from all self-data are retained. This is a **hard mathematical
constraint**, not a probability threshold.

**Verified by**: `tests/test_negsel.py::test_detectors_never_fire_on_self`

**False Positive Rate**: 0.0000 on self-data (by construction, not by tuning).

### 2. Cryptographic Integrity of Agent State

**Guarantee**: Signed agent state files will be **rejected** if modified after signing.

**Mechanism**: Ed25519 digital signatures (with HMAC-SHA256 fallback) on serialized agent
state. The `SecureVerifier` computes SHA-256 hashes of file contents and verifies
signatures before loading.

**Verified by**: `tests/test_signer_verifier.py::test_tampered_data_rejected`,
`test_verify_tampered_file_fails`, `test_load_tampered_raises`

### 3. Deterministic Feature Extraction

**Guarantee**: Given the same input text/code, the Dendritic Agent will **always** produce
the **identical** 20-dimensional feature vector.

**Mechanism**: Rule-based regex pattern matching and word counting. No randomness, no
model inference, no external API calls.

**Verified by**: `tests/test_dendritic_determinism.py::test_same_input_same_vector_100_runs`

## What Antigence Does NOT Guarantee

### 1. Detection of All Anomalies

The NegSl-AIS algorithm guarantees zero false positives on self-data, but it does **not**
guarantee detection of all non-self data. The recall (true positive rate) depends on:
- The quality and coverage of the training data
- The `r_self` parameter (larger values = more conservative detection)
- The dimensionality and distribution of the feature space

**Current benchmark**: 35% recall on synthetic code security data with Dendritic features.
This will improve with embedding-based features.

### 2. Semantic Understanding

The Dendritic Agent uses regex-based feature extraction, not semantic analysis. It can
detect structural patterns (citations, hedging words, danger keywords) but does **not**
understand meaning, context, or intent.

### 3. Protection Against All LLM Failure Modes

Antigence currently detects:
- Code patterns associated with known vulnerability classes
- Text patterns associated with misinformation (danger signals)
- Anomalous inputs that differ from training distribution

Antigence does **not** currently detect:
- Subtle logical errors in code
- Factually incorrect but structurally normal text
- Novel attack vectors not represented in the feature space
- Social engineering or manipulation

### 4. Real-Time Performance at Scale

Current implementation is optimized for correctness, not throughput. Batch processing of
thousands of samples per second is possible with PyTorch acceleration, but real-time
latency guarantees are not provided.

## Mathematical Foundation

- **NegSl-AIS**: Umair et al. (2025), "Negative selection-based artificial immune system",
  Results in Engineering 27, 106601.
- **Opt-AiNet**: de Castro & Timmis (2002), immune network optimization.
- **Affinity**: Immunos-81 formula with exponential decay and cosine similarity.

## Verification

All guarantees are tested in the automated test suite:

```bash
pytest tests/ -v --ignore=tests/smoke_test.py
```

83 tests verify the claims above. If any test fails, the corresponding guarantee
is invalidated and must be investigated before release.
