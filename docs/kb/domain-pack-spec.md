# Domain Pack Specification (immunOS)

## Purpose
Define a portable, domain-scoped configuration that binds antigen presentation, receptor libraries, thresholds, and hardware tiers for a specific use case.

## Core Idea
A domain pack is the deployable unit of "immune specialization" in immunOS. It defines:
- which features are extracted (antigen presentation),
- which receptors are active (NK/B/T memory),
- how thresholds are tuned,
- and which hardware tier is supported.

## Required Fields
- `id`: unique identifier (e.g., research-v1)
- `domain`: use case (research, hallucination, network, code, compliance)
- `version`: semantic version
- `features`: list of feature extractors and inputs
- `receptors`: NK/B/T configuration per domain
- `thresholds`: default decision thresholds per receptor
- `hardware_tiers`: allowed compute tiers and model mappings
- `data_sources`: training and evaluation datasets
- `metrics`: evaluation metrics to report
- `policies`: safety or compliance rules

## Optional Fields
- `notes`: operational guidance
- `owner`: maintainer or team
- `dependencies`: required models or tools

## Example (YAML)
```yaml
id: research-v1
name: Research Verification Pack
domain: research
version: 1.0.0
features:
  - name: dendritic_claim_features
    inputs: [claim_text, evidence_sentences, citations]
receptors:
  nk:
    detector_set: nk_research_v1.json
    mode: feature
  bcell:
    classifier: bcell_research_v1.pt
    labels: [support, contradict, nei]
  tcell:
    memory_store: tcell_research_v1
thresholds:
  nk: 0.95
  bcell: 0.80
hardware_tiers:
  tier0_airgapped:
    models: [qwen2.5:1.5b]
    max_memory_gb: 8
  tier1_laptop:
    models: [qwen2.5:7b]
    max_memory_gb: 32
  tier2_workstation:
    models: [deepseek-r1:14b]
    max_memory_gb: 128
  tier3_cloud:
    models: [claude-opus-4.5]
    max_memory_gb: 512
data_sources:
  train: [SciFact]
  eval: [SciFact-dev]
metrics:
  - sentence_selection_f1
  - abstract_label_f1
policies:
  - require_evidence_links
  - block_if_no_sources
```

## Notes
- Domain packs should not mix unrelated feature spaces.
- Cross-domain reuse is allowed only for coarse triage.
- Pack versions should be updated when thresholds or detectors change.
- Initial packs are stored in `immunos-mcp/config/domain-packs/`.

---

**Last Updated**: 2025-12-26
**Maintained by**: IMMUNOS System
