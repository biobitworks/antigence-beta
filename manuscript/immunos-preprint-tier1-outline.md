---
title: "immunOS Tier 1 Outline: Baseline Evaluation Plan"
author:
  - name: Byron
    affiliation: BiobitWorks
    email: byron@biobitworks.com
date: 2025-12-29
version: v0.1-outline
status: tier1-outline
license: CC-BY-4.0
repository: https://github.com/biobitworks/immunos
---

# immunOS Tier 1 Outline: Baseline Evaluation Plan

## Purpose

Tier 1 provides baseline evaluation results for claim verification and anomaly detection. This
outline enumerates datasets, metrics, and run protocols to ensure reproducibility.

## Core Questions

1. Can immunOS provide competitive baseline performance on scientific claim verification?
2. Do NK-style negative selection detectors provide measurable anomaly detection signals?
3. Are results reproducible across datasets and evidence modes?

## Datasets (Priority Order)

### Research Claim Verification
- SciFact (claims + evidence sentences)
- FEVER (claims + evidence sentences)
- FEVEROUS (claims + evidence with Wikipedia pages)
- HoVer (claims + evidence; note missing wiki pages in some setups)

### Tabular / Structured Evidence
- TabFact (claims + tables)

### Textual Hallucination / Refutation
- VitaminC (claims + evidence)

## Evidence Modes

- Claim-only (baseline, no evidence)
- Evidence retrieval (embedding or keyword)
- Gold evidence (where available)

## Metrics

- Precision, recall, F1
- Accuracy (when labels are balanced)
- Per-dataset breakdowns
- Threshold sensitivity (NK detector threshold sweeps)

## Planned Runs

1. SciFact baseline (open retrieval, dev set)
2. FEVER baseline (retrieval evidence, k=10)
3. FEVEROUS gold evidence
4. HoVer claim-only (until wiki pages are available)
5. TabFact evidence mode
6. VitaminC evidence mode

## Artifacts and Logging

- Store outputs in immunos-preprint/publication/data/
- Store notes in immunos-preprint/publication/notes/
- Log metrics in .immunos/TRAINING_METRICS.md
- Record run manifests and parameters for reproducibility

## Planned Sensitivity Analyses

- NK threshold sweeps (precision vs recall trade-off)
- Embedding backend comparisons (simple_text vs bge-m3)
- Retrieval parameter sweeps (k, max-per-page)

## Exit Criteria for Tier 1

- At least one complete baseline per dataset family
- Reproducible run scripts and logs
- Documented limitations and missing data

