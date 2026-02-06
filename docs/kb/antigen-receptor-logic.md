# Antigen Presentation and Receptor Logic (immunOS)

## Purpose
Define how immunOS presents "antigens" (inputs) and routes them to domain-appropriate receptors (detectors) on the correct cell types.

## Core Principles
1) Domain-specific receptors
- Each use case has its own receptor library (research, hallucination, network, code, etc.).
- Receptors are trained on domain features and should not be shared across unrelated domains.

2) Antigen presentation is feature-first
- Raw inputs are converted into a structured antigen representation by Dendritic feature extraction.
- Receptors operate on feature vectors, not raw text or logs.

3) Hardware-aware tiers
- Receptors are deployed in tiers based on available compute.
- Higher tiers allow deeper models and tighter thresholds; lower tiers use smaller detectors with conservative gating.

4) Multi-stage decisions
- NK cells provide fast anomaly screening.
- B cells provide label verification and evidence checks.
- T cell memory stabilizes repeated patterns and reduces false positives.

## Antigen Presentation Pipeline

1) Intake
- Input arrives from a domain source (claims, code diff, network flow, etc.).
- Lightweight guards (sanitization, PII filters) apply at the boundary.

2) Domain routing (Lymph Node)
- Assign a domain based on source and content.
- Example domains: research, hallucination, network, code, compliance.

3) Dendritic presentation
- Extract a domain-specific feature vector.
- Package as an antigen object with metadata (domain, source, timestamp, confidence).

4) Receptor evaluation
- NK receptors: negative-selection detector vectors flag non-self/anomalous inputs.
- B cell receptors: classifiers label support/contradict/NEI or safe/vulnerable.
- Thresholds are tuned per domain and risk profile.

5) Decision + memory
- Produce a self/non-self decision with evidence trace.
- Store validated patterns for recall in T cell memory.

## Receptor Types (immunOS Mapping)

- NK receptors: detector vectors in the domain feature space; fast anomaly checks.
- B cell receptors: trained classifiers for verification/labeling tasks.
- Dendritic presentation: feature extraction and antigen packaging.
- T cell memory: stored validated patterns used for stabilization and recall.

## Domain-Specific Receptor Libraries

Each domain maintains its own receptor set:

- Research verification
  - Features: evidence similarity, citation presence, hedging, contradictions.
  - NK: detects anomalous claims outside evidence distribution.
  - B cell: support/contradict/NEI labeler.

- Hallucination control
  - Features: hedging, uncertainty markers, refusal patterns, groundedness signals.
  - NK: flags non-grounded responses at high thresholds.
  - B cell: truthful vs hallucinated classification.

- Network security (NSL-KDD)
  - Features: protocol, duration, bytes, flags, error rates, host stats.
  - NK: detects attack-like anomalies after threshold calibration.

- Code security
  - Features: unsafe API usage, complexity, taint patterns, dependency risk.
  - NK: anomaly detection for suspicious code patterns.
  - B cell: safe vs vulnerable classifier.

## Domain Packs
- Domain packs are the configuration artifact that binds features, receptors, thresholds, and hardware tiers.
- See: `docs/kb/domain-pack-spec.md` and `templates/domain-pack.yaml`.

## Hardware-Aware Tiers

- Tier 0 (airgapped / low power)
  - Minimal detectors, conservative thresholds, NK-only triage.

- Tier 1 (laptop / small GPU)
  - NK + B cell with small models, limited feature sets.

- Tier 2 (workstation / mid GPU)
  - Full Dendritic features, tuned NK thresholds, B cell verification.

- Tier 3 (cloud / H100 class)
  - Full pipeline with ensembles, calibration sweeps, and audit trails.

## Training and Lifecycle

1) Receptor genesis
- Initialize detectors with domain-specific feature ranges.

2) Negative selection
- Remove receptors that match self data too often.

3) Calibration
- Tune thresholds for desired precision/recall trade-offs.

4) Validation
- Evaluate on held-out sets with domain metrics.

5) Memory + rotation
- Store validated patterns; retire stale or overfitting receptors.

## Implementation Anchors
- Antigen object: `immunos-mcp/src/immunos_mcp/core/antigen.py`
- Training pipeline: `immunos-mcp/src/immunos_mcp/training/`
- Agents: `immunos-mcp/src/immunos_mcp/agents/`

## Notes
- This design mirrors biological antigen presentation: pathogens present antigens; receptors on specialized cells recognize domain-specific patterns.
- Cross-domain receptor reuse should be avoided except for coarse triage.

---

**Last Updated**: 2025-12-26
**Maintained by**: IMMUNOS System
