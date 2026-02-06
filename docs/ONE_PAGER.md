# Antigence (Open‑Source + Enterprise)

## What it is
Antigence is an open‑source, local‑first, bio‑inspired security analysis platform based on Artificial Immune System (AIS) principles. It combines adaptive pattern recognition (B Cells) with negative selection (NK Cells) to detect anomalies, hallucinations, and security threats in code and text.

## Why it matters
Most AI security tools assume centralized data. Antigence is designed to run **locally** so organizations can analyze sensitive data **without uploading it**.

## Key capabilities
- Multi‑agent analysis (B Cell, NK Cell, Dendritic, Memory, Orchestrator)
- Local‑first: no telemetry by default
- Extensible: domain packs and custom training
- Works with public datasets; no proprietary data shipped

## Open source + ROI
Antigence is **open source** with an **open‑core** model for enterprise:
- Hosted deployments
- SLAs and onboarding
- Compliance tooling and audit trails
- Enterprise features (SSO, centralized policy)

## Public data policy
No proprietary or user‑derived data is included in the public repo. Public datasets can be downloaded using built‑in scripts.

## Quick start
```bash
pip install -e .
python scripts/download_datasets.py
```

## Repo
https://github.com/biobitworks/antigence-beta

## Contact
enterprise@biobitworks.com
