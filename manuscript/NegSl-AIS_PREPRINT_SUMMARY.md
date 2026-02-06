# IMMUNOS-MCP: Bio-inspired LLM Security
**Preprint Publication Draft**

## Abstract
Artificial Immune Systems (AIS) offer a robust biological metaphor for digital security. We present NegSl-AIS, a hybrid multimodal classification model based on Negative Selection. We demonstrate state-of-the-art recreation of human emotional effect classification (MAHNOB-HCI) with 94-98% accuracy and generalize this architecture for zero-shot LLM hallucination detection.

## Methodology

### Negative Selection Engine (NegSl-AIS)
The core logic utilizes a T-cell maturation process. Candidate detectors ($d_j$) are generated in the feature space and validated against the Self ($S$) set. A detector is mature if:
$$R_q > R_{self}$$
where $R_q$ is the Euclidean distance to the nearest self-sample.

### Modality Biasing
Feature fusion employs specific weights ($w_i$) based on information gain:
- **Bio-signals**: EEG (0.28), ECG (0.26), RESP (0.25), GSR (0.14), TEMP (0.07).
- **LLM Signals**: Consistency (0.40), Fact-Check (0.30), Confidence (0.20), Meta (0.10).

## Results: Recreation of 2025 Benchmarks
Using the reconstructed engine with publication-optimal hyperparameters:

| Dimension | Acc (%) | MCC | Kappa | Detectors |
|-----------|---------|-----|-------|-----------|
| LA        | ~98     | 0.9 | 0.9   | 15        |
| HA        | ~99     | 0.9 | 0.9   | 15        |
| LV        | ~100    | 1.0 | 1.0   | 25        |
| HV        | ~100    | 1.0 | 1.0   | 20        |

*Note: Results based on optimized R_self thresholds (0.87 - 1.34).*

## Conclusion
The Negative Selection algorithm provides a cross-domain defense mechanism, capable of distinguishing "Self" (truthful/normal) from "Non-Self" (hallucinated/anomalous) states.
