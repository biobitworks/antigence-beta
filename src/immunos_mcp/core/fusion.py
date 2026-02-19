"""
NegSl-AIS Modality Biasing & Feature Fusion logic.
Based on Paper weights: EEG: 0.28, ECG: 0.26, RESP: 0.25, GSR: 0.14, TEMP: 0.07

Extended with ImmuneSignalFusion for the Affinity-First Architecture:
Fuses NegSel antibody bindings + dendritic features + danger signals
into a single binding strength → ImmuneResponse.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from .immune_response import ImmuneResponse, CITATION_THRESHOLDS, ANALYSIS_THRESHOLDS, SECURITY_THRESHOLDS

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


# ---------------------------------------------------------------------------
# Immune Signal Fusion — Affinity-First Architecture
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """Result from immune signal fusion."""
    fused_binding: float      # Combined binding strength [0, 1]
    response: ImmuneResponse  # IGNORE / REVIEW / REJECT
    antibody_aggregate: float # Aggregated antibody binding
    dendritic_score: float    # Dendritic cell feature score
    danger_signal: float      # Context-dependent danger signal


# Domain-specific signal weights
DOMAIN_WEIGHTS = {
    "citation": {"antibody": 0.50, "dendritic": 0.30, "danger": 0.20},
    "analysis": {"antibody": 0.45, "dendritic": 0.35, "danger": 0.20},
    "security": {"antibody": 0.55, "dendritic": 0.15, "danger": 0.30},
}

# Domain-specific response thresholds
DOMAIN_THRESHOLDS = {
    "citation": CITATION_THRESHOLDS,
    "analysis": ANALYSIS_THRESHOLDS,
    "security": SECURITY_THRESHOLDS,
}


class ImmuneSignalFusion:
    """
    Fuses three immune signal types into a single binding strength.

    Signal types:
    1. Antibody bindings — NegSel affinity scores from domain antibodies
    2. Dendritic score — context/feature extraction confidence
    3. Danger signal — external context (first seen, source trust, etc.)

    Aggregation:
    - Antibody bindings: 70% mean + 30% max (catch worst-case outlier)
    - Final: weighted sum of antibody_agg, dendritic, danger
    - Maps to ImmuneResponse via domain-specific thresholds
    """

    def __init__(self, domain: str = "citation"):
        if domain not in DOMAIN_WEIGHTS:
            raise ValueError(f"Unknown domain: {domain}. Valid: {list(DOMAIN_WEIGHTS.keys())}")
        self.domain = domain
        self.weights = DOMAIN_WEIGHTS[domain]
        self.thresholds = DOMAIN_THRESHOLDS[domain]

    def aggregate_antibody_bindings(self, bindings: List[float]) -> float:
        """Aggregate multiple antibody bindings: 70% mean + 30% max."""
        if not bindings:
            return 0.0
        mean_binding = sum(bindings) / len(bindings)
        max_binding = max(bindings)
        return 0.7 * mean_binding + 0.3 * max_binding

    def fuse_signals(
        self,
        antibody_bindings: List[float],
        dendritic_score: float = 0.0,
        danger_signal: float = 0.0,
    ) -> FusionResult:
        """
        Fuse immune signals into a single binding strength and response.

        Weight normalization: when signals are missing (score=0.0 and not
        explicitly provided), their weight is redistributed to active signals.
        This prevents the fused score from being artificially capped when only
        antibody bindings are available (EXP-001 finding: antibody-only fusion
        must use the full [0,1] range to hit REJECT thresholds).

        Args:
            antibody_bindings: List of binding affinities from domain antibodies [0,1]
            dendritic_score: Dendritic cell feature confidence [0,1]
            danger_signal: Context-dependent danger signal [0,1]

        Returns:
            FusionResult with fused binding, immune response, and component scores
        """
        # Aggregate antibody bindings
        ab_agg = self.aggregate_antibody_bindings(antibody_bindings)

        # Clamp inputs to [0, 1]
        ab_agg = max(0.0, min(1.0, ab_agg))
        dendritic_score = max(0.0, min(1.0, dendritic_score))
        danger_signal = max(0.0, min(1.0, danger_signal))

        # Normalize weights to active signals only.
        # Inactive signals (0.0) donate their weight proportionally.
        active = {}
        if antibody_bindings:
            active["antibody"] = self.weights["antibody"]
        if dendritic_score > 0.0:
            active["dendritic"] = self.weights["dendritic"]
        if danger_signal > 0.0:
            active["danger"] = self.weights["danger"]

        total_active = sum(active.values()) if active else 1.0

        # Weighted fusion with normalized weights
        fused = 0.0
        if "antibody" in active:
            fused += (active["antibody"] / total_active) * ab_agg
        if "dendritic" in active:
            fused += (active["dendritic"] / total_active) * dendritic_score
        if "danger" in active:
            fused += (active["danger"] / total_active) * danger_signal
        fused = max(0.0, min(1.0, fused))

        # Classify via domain thresholds
        response = self.thresholds.classify(fused)

        return FusionResult(
            fused_binding=fused,
            response=response,
            antibody_aggregate=ab_agg,
            dendritic_score=dendritic_score,
            danger_signal=danger_signal,
        )
