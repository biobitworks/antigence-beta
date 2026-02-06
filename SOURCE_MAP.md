# Source Map: Antigence Alpha (v0.1.0-alpha)

This document classifies the project files into **Public** (Zenodo/bioRxiv ready) and **Private** (Internal Research/Data) categories.

## 1. Public Files (Open Research Core)
*All code implementing the AIS logic and general safety architecture.*
- `src/immunos_mcp/algorithms/negsel.py`: Core logic for Eq 20-22.
- `src/immunos_mcp/core/fusion.py`: Modality biasing weights.
- `src/immunos_mcp/agents/citation_detector.py`: Cross-domain transition code.
- `scripts/reproduce_negsl_ais.py`: Reproducibility script.
- `manuscript/*.md`: Preprint drafts and summaries.
- `web_app/`: Frontend interface for demonstration.
- `CITATIONS.bib` / `CITATION.cff`: Meta-documentation.

## 2. Private Files (DO NOT PUSH PUBLIC)
*Files containing sensitive keys, proprietary research notes, or experimental raw data.*
- `antigence-alpha/context.json`: Internal agent context and history.
- `web_app/data/immunos.db`: Local analysis history and user submissions.
- `antigence-alpha/blog/`: Unreleased thinking and substack drafts.
- `docs/hallucination_data_sources.md`: Private data gathering targets.
- `docs/HALLUCINATION_DATA_SOURCES.md`: Scraper targets and IP.

## 3. Data Sensitivity Policy
The **MAHNOB-HCI** dataset is subject to its original license from Imperial College London. It is **NOT** included in this repository. Researchers must obtain it directly from the source to run the full bio-signal benchmarks.

## 4. Maintenance
- **.gitignore**: Updated to exclude `*.db`, `context.json`, and all `runs/` directories.
- **Snapshot Integrity**: Final Tier 1 implementations are locked in `src/immunos_mcp/`.
