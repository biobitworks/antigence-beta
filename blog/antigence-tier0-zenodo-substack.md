# immunOS Tier 0 (Antigence™ pending): Hypothesis + Software Overview

Today I’m releasing **immunOS Tier 0**, a local-first artificial immune system (AIS) for research
integrity and AI safety. Tier 0 is intentionally light on results. It is a citable software overview
that states the hypothesis, defines the immune-inspired architecture, and locks the terminology
before the data-heavy preprint lands.

## Why a Tier 0 release

AI tools can hallucinate, mis-cite, or drift from the source of truth. We need a local, auditable
safety layer that works offline in sensitive research settings. immunOS maps biological immune logic
to software: self vs non-self detection, multi-agent responses, and memory of verified patterns.

Tier 0 keeps it clean and citable:
- Hypothesis only, no benchmark claims
- Architecture overview and agent roles
- Terminology and safety constraints

## Core idea

An immune system does not generate truths. It inspects inputs against a trusted definition of self
and raises alerts when patterns look foreign or unsafe. immunOS applies the same logic to research
integrity: claim verification, anomaly detection, and provenance checks across AI outputs.

## Terminology (short version)

- **antigent**: a single immune-style agent instance
- **antigents**: a set of agents in a domain pack
- **Antigence™ (TM pending)**: the platform brand identity
- **antigentic**: workflows designed around self vs non-self detection

## What is next

Tier 1 will add baseline evaluation results across claim verification datasets. Tier 2 will extend
comparative analysis and deployment guidance. Everything stays local-first and reproducible.

If you want to follow the work or contribute, the repository is public:
https://github.com/biobitworks/immunos
