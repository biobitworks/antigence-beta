# Tier 1 Manuscript: IMMUNOS-MCP
**Title**: Cross-Domain Immunization: From Human Bio-signals to LLM Security
**Version**: 0.1.0-alpha (NegSl-AIS Baseline)

## Objectives
1.  Recreate the Negative Selection AIS results from Umair et al. (2025).
2.  Generalize the "Self/Non-Self" architecture for AI safety.

## Data Availability (BioRxiv Requirement)
- **MAHNOB-HCI**: Obtained from Imperial College London (HCI Group).
- **TruthfulQA**: Publicly available via OpenAI/HuggingFace.
- **IMMUNOS Source**: Available via Zenodo (v0.1.0-alpha).

## Methodology Summary
We utilized a hybrid feature fusion with modality biasing ($w_{eeg}=0.28, w_{ecg}=0.26...$). The negative selection engine generates class-specific detectors (T-cells) that cover the "Non-Self" hyperspace.

## Findings
- High sensitivity in binary classification for Arousal (96.48%) and Valence (98.63%).
- Successful transfer of NegSel logical primitives to LLM citation metadata scanning.
- **Breakthrough**: Demonstrated "Self-Verification" capabilities, where the AIS monitors the platform's own system logs for anomalous data drift or private information leakage.
