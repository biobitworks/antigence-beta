# Changelog

## v0.3.0-beta — 2026-02-19

### Synced from antigence-alpha commit ad641f7

#### Added: 4 New Antibody Systems (19 antibodies, 118 tests)

Antibody inventory: 6 → 25 across 5 systems. Full test suite: 218 passed, 0 failures.

- **MethodologyAntibodySystem** (6): SampleSize, Blinding, ControlGroup, Preregistration, Randomization, InclusionCriteria
- **FigureIntegrityAntibodySystem** (6): CaptionCompleteness, RepresentativeClaim, Quantification, PanelConsistency, DataPresentation, ImageSource
- **CitationPatternAntibodySystem** (3): Retraction, SelfCitation, PredatoryJournal
- **PromptInjectionAntibodySystem** (4): IndirectInjection, Jailbreak, EncodingEvasion, ContextOverflow

#### Added: Core dependencies (ported from alpha)
- `src/immunos_mcp/core/immune_response.py` — ImmuneResponse enum, ResponseThresholds
- `src/immunos_mcp/core/fusion.py` — ImmuneSignalFusion (appended to existing ModalityFusion)
- `src/immunos_mcp/algorithms/negsel.py` — adaptive r_self, score calibration (backwards-compatible)

#### Wired
- MCP server imports + instantiates all 4 new systems

## v0.1.0-alpha — 2026-01-05
- **NegSl-AIS Integration**: Mathematically rigorous implementation of the 2025 Negative Selection Artificial Immune System for emotion classification.
- **Modality Biasing**: Implemented bio-signal weighting ($0.28/0.26/0.25/0.14/0.07$) for hybrid feature fusion.
- **LLM Security Foundation**: Extended NegSl-AIS logic to LLM hallucination detection (TruthfulQA context).
- **Publication Readiness**: Added `reproduce_negsl_ais.py` and `NegSl-AIS_PREPRINT_SUMMARY.md` for bioRxiv preprint (Tier 1).
- **Automated Reporting**: Added `generate_preprint_report.py` for results verification.

## v0.0-alpha — 2025-12-29
- Initial Antigence Alpha snapshot (Tier 0 manuscript, Tier 1 outline, branding, lexicon, pitch, blog draft).
