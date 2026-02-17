"""
NegSl-AIS Modality Biasing & Feature Fusion logic.
Based on Paper weights: EEG: 0.28, ECG: 0.26, RESP: 0.25, GSR: 0.14, TEMP: 0.07
"""

import numpy as np
from typing import Dict, Optional

class ModalityFusion:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Table 4 Bias Weights
        self.weights = weights or {
            "EEG": 0.28,
            "ECG": 0.26,
            "RESP": 0.25,
            "GSR": 0.14,
            "TEMP": 0.07
        }
        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def fuse(self, modality_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Perform hybrid feature fusion with modality biasing.
        """
        fused = []
        for modality, feature_vec in modality_features.items():
            if modality in self.weights:
                # Apply bias (biasing can be used to scale or just concatenated)
                # Paper: "modality biasing prior to feature fusion"
                biased_vec = feature_vec * self.weights[modality]
                fused.append(biased_vec)

        return np.concatenate(fused) if fused else np.array([])

class LLMModalityFusion(ModalityFusion):
    """
    Bio-inspired fusion applied to LLM safety signals.
    """
    def __init__(self):
        weights = {
            "CONSISTENCY": 0.40,  # Self-consistency (Self)
            "FACT_CHECK": 0.30,   # External knowledge
            "CONFIDENCE": 0.20,   # Model probability
            "METADATA": 0.10      # Syntax/Structure
        }
        super().__init__(weights)
