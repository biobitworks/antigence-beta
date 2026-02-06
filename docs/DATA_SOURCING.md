# Antigence™ Data Sourcing Guide

This document outlines the recommended datasets for scaling Antigence™ training beyond the initial alpha release.

## 1. Network Anomaly Detection (NK-Cells)
To build robust "Self" models for network traffic, we use:
- **NSL-KDD**: Benchmark dataset for intrusion detection.
  - *Source*: [UNB NSL-KDD](https://www.unb.ca/cic/datasets/nsl-kdd.html)
  - *Usage*: Train NK-Cells on "Normal" traffic patterns to detect zero-day intrusions.
- **CIC-IDS2017**: Modern network traffic flows.
  - *Source*: [UNB CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

## 2. Code & Web Security (B-Cells / NK-Cells)
For pattern recognition of known vulnerabilities:
- **SQLiV5**: Expanded SQL injection dataset.
  - *Source*: [GitHub: nidnogg/sqliv5-dataset](https://github.com/nidnogg/sqliv5-dataset)
- **Malicious URL Dataset**: 650k+ labeled URLs.
  - *Source*: [Kaggle Malicious Phish](https://www.kaggle.com/datasets/sid321ss/malicious-urls-dataset)

## 3. Multimodal Emotion Recognition (NK-Cells)
As an alternative to MAHNOB-HCI:
- **DEAP**: Emotion analysis using physiological signals.
  - *Source*: [DEAP Dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
- **AMIGOS**: Individual and group emotional experiences.
  - *Source*: [AMIGOS Dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/)

## Implementation Notes
Users should download CSV versions of these datasets into `data/training/` and use the provided `src/immunos_mcp/scripts/train_agent.py` (coming soon) to generate detectors.
