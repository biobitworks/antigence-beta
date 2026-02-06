---
title: "immunOS Tier 0: Hypothesis and Software Overview"
author:
  - name: Byron
    affiliation: BiobitWorks
    email: byron@biobitworks.com
date: 2025-12-29
version: v0.0
status: tier0-hypothesis
license: CC-BY-4.0
repository: https://github.com/biobitworks/immunos
---

# immunOS Tier 0: Hypothesis and Software Overview

**Author**: Byron (BiobitWorks)
**Date**: December 2025
**Version**: 0.0 (Tier 0)

## Abstract

immunOS is a local-first verification system that applies artificial immune system (AIS) principles to
research integrity and safety. The core hypothesis is that self versus non-self discrimination can be
implemented as a multi-agent immune architecture that detects hallucinations, unsupported claims, and
incoherent reasoning in scientific workflows. This Tier 0 document introduces the hypothesis, key
concepts, and software architecture without reporting results. It is intended as a citable software
overview for early dissemination (e.g., Zenodo) ahead of a full preprint with evaluation data.
No empirical results are reported in this Tier 0 release.

**Keywords**: artificial immune system, research integrity, claim verification, negative selection, local AI

---

## 1. Hypothesis

A biological immune system maintains organismal integrity by detecting non-self patterns and
coordinating multi-cell responses. We hypothesize that a software analog can provide a safety layer for
AI-enabled research by:

1. Defining a trusted "self" corpus (claims, evidence, protocols, known-good code).
2. Generating detectors that do not match self (negative selection).
3. Routing inputs through specialized immune-style agents to score risk and provenance.
4. Persisting memory of validated and rejected patterns to improve future detection.

If successful, this yields a local, auditable mechanism for claim verification, tool gating, and
research integrity checks that does not depend on external APIs.

## 2. Software Overview

immunOS is composed of immune-inspired agents ("antigents") coordinated by an orchestrator.

- **B-Cell antigents**: pattern recognition and claim classification
- **NK-Cell antigents**: anomaly detection via negative selection
- **Dendritic antigents**: feature extraction and signal derivation
- **T-Cell antigents**: coordination and context integration
- **Memory antigents**: persistence of verified patterns

The system is designed to run locally with pluggable models (e.g., Ollama), supporting air-gapped
research environments and reproducible evaluation.

### Terminology

- **antigent**: antigen + agent, a single immune-style agent instance
- **antigents**: the set of immune agents in a run
- **Antigence™ (TM pending)**: platform brand identity (research identity layer)
- **antigentic**: workflows designed around self/non-self detection

## 3. Architecture (Tier 0)

```
Input -> Orchestrator
         |-> Dendritic (features)
         |-> B-Cell (pattern match)
         |-> NK-Cell (anomaly score)
         |-> Memory (persist)
Output -> Structured verdict + provenance
```

Key properties:

- Local-first execution
- Modular agent roles
- Negative selection detector training
- Audit-friendly logs and reproducible configs

## 4. Scope and Limitations

Tier 0 is intentionally limited to hypothesis and software description. It does not report
benchmark results, training metrics, or claims of performance. Full evaluation is planned for Tier 1
and Tier 2 preprints.

## 5. Roadmap

- **Tier 0**: hypothesis + software overview (this document)
- **Tier 1**: baseline evaluation on claim verification datasets
- **Tier 2**: extended evaluation + comparative analysis + deployment guidance

## 6. Data and Availability

- Codebase: https://github.com/biobitworks/immunos
- Data, logs, and artifacts will be published in Tier 1 and Tier 2 releases.

## 7. Ethics and Safety

immunOS is designed as a safety layer. It does not automate scientific conclusions and should be used
as decision support. All outputs are intended to be auditable and reviewable by humans.

---

## Acknowledgments

This early software overview was compiled with local tools and internal documentation. No external
APIs were required for Tier 0 preparation.
