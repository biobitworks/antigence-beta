# Citation Immunity: Architecture for LLM Agents

## The Challenge
Detecting hallucinated citations without a $100\%$ complete database of all scholarly works.

## 1. Where do the Antibodies Live?
We position the immune system in two distinct layers:

### A. Inside the Black Box (Self-Consistency / Lymphocytes)
- **Role**: Detects internal contradictions before the "thought" is externalized.
- **Detector Type**: Probabilistic Antibodies.
- **Logic**: If the LLM generates a citation for a fact, and then fails to reproduce the same citation components (DOI, Author, Journal) under slight perturbation (sampling variation), it has failed the "Self" consistency test.

### B. Outside the Black Box (System Interface / Dendritic Cells)
- **Role**: Scans output structure and metadata against "Hallucination Archetypes."
- **Detector Type**: Multi-modal Detectors.
- **Logic**: Scanning for "Non-Self" patterns. Hallucinated citations often follow high-entropy patterns or "hallucination archetypes" (e.g., matching a famous author to a plausible-but-fake title, or generating DOIs with invalid checksum patterns).

## 2. NegSl-AIS Citation Logic
We treat valid citation patterns as **Self** and anomalous metadata patterns as **Non-Self**.

### Detectors Needed:
1.  **Semantic Cross-Check (Internal)**: Does the paper title actually align with the retrieved abstract?
2.  **Metadata Entropy (External)**: Hallucinated DOIs/URLs often have higher character entropy or invalid nested structures.
3.  **Journal/Author Collision**: Flagging when an author's known research domain is ≥3 standard deviations away from the paper's topic (unless verified).

## 3. Implementation: `CitationAnomalyDetector`
This detector lives at the **Agent Interface**. It stops the agent from "releasing" a citation to the user if it binds to a "hallucination detector."

```python
# Prototype logic for Citation Immunization
# 1. 'Self' is a set of known valid citation formats and author/journal relationships.
# 2. 'Detectors' (antibodies) are trained to bind to:
#    - Invalid DOI patterns
#    - Topic-Author domain mismatches
#    - High-entropy 'noise' titles
```
