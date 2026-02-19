#!/usr/bin/env python3
"""
IMMUNOS Negative Selection Module
==================================

Implements the NegSl-AIS algorithm for emotion classification and general anomaly detection.
Based on: Muhammad Umair et al. (2025) - "Negative selection-based artificial immune system (NegSl-AIS)"
Published in Results in Engineering 27 (2025) 106601.

Core Concept:
- Maturation of T-cells (detectors) in feature space.
- Binding rule: R_q > R_self for valid detectors.
- Class-specific parameters optimal for MAHNOB-HCI recreation.
"""

import numpy as np
import os
from typing import List, Optional, Union
from dataclasses import dataclass

# =============================================================================
# PAPER PRESETS (Table 5 & 6)
# =============================================================================

@dataclass
class NegSelConfig:
    """Configuration for NegSl-AIS detectors based on publication parameters."""
    num_detectors: int
    r_self: float
    description: str = ""
    adaptive: bool = False  # Compute r_self from self-data pairwise distances

# Paper Optimal Parameters
NEGSEL_PRESETS = {
    "LA": NegSelConfig(num_detectors=15, r_self=0.87, description="Low Arousal"),
    "HA": NegSelConfig(num_detectors=15, r_self=0.91, description="High Arousal"),
    "LV": NegSelConfig(num_detectors=25, r_self=1.31, description="Low Valence"),
    "HV": NegSelConfig(num_detectors=20, r_self=1.33, description="High Valence"),
    "LLM_HALLUCINATION": NegSelConfig(num_detectors=50, r_self=0.15, description="LLM Security Baseline"),
    "GENERAL": NegSelConfig(num_detectors=20, r_self=0.85, description="General Anomaly")
}

# =============================================================================
# MODALITY BIASING (Section 4.2.19, Table 4)
# =============================================================================

MODALITY_WEIGHTS_PAPER = {
    "EEG": 0.28,
    "ECG": 0.26,
    "RESP": 0.25,
    "GSR": 0.14,
    "TEMP": 0.07
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Detector:
    """
    Represents a valid detector that passed negative selection.
    Equation 20: Detector = {Valid if R_q > R_self}
    """
    center: np.ndarray
    radius: float       # Paper: r^j = R_q - R_self
    class_label: str
    r_self: float

    def __post_init__(self):
        if self.radius < 0:
             # In some paper variations, invalid detectors aren't discarded but here they must be >=0
             pass


# =============================================================================
# NEGATIVE SELECTION CLASSIFIER
# =============================================================================

class NegativeSelectionClassifier:
    """
    NegSl-AIS implementation following Umair et al. (2025).
    """

    def __init__(self, config: Union[NegSelConfig, str] = "GENERAL",
                 class_label: str = "SELF", db_path: str = None):
        if isinstance(config, str):
            self.config = NEGSEL_PRESETS.get(config, NEGSEL_PRESETS["GENERAL"])
        else:
            self.config = config

        self.class_label = class_label
        self.db_path = db_path or os.getenv("IMMUNOS_DB") or os.path.join(os.path.expanduser("~"), ".immunos", "db", "immunos.db")

        self.valid_detectors: List[Detector] = []
        self.feature_dim: int = 0
        self.self_samples: Optional[np.ndarray] = None
        # Score calibration (populated during fit)
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._effective_r_self: float = self.config.r_self

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Eq 21: R_q = sqrt(sum((Omega_k - d_k)^2))"""
        return np.sqrt(np.sum((a - b) ** 2))

    def _get_nearest_self_distance(self, candidate: np.ndarray) -> float:
        """R_q = min(||Self_i, d_j||)"""
        distances = np.linalg.norm(self.self_samples - candidate, axis=1)
        return np.min(distances)

    def _compute_adaptive_r_self(self, self_samples: np.ndarray) -> float:
        """Compute r_self from the self-data pairwise distance distribution.

        Uses the 95th percentile of nearest-neighbor distances among self
        samples. This ensures nearly all self-samples score 0 (within r_self
        of their nearest neighbor), while setting the boundary just beyond
        the natural self-cluster extent.
        """
        n = self_samples.shape[0]
        if n < 2:
            return self.config.r_self

        # Compute pairwise distance matrix
        # For N samples, this is O(N^2) but N is typically 10-40
        dists = np.linalg.norm(
            self_samples[:, np.newaxis] - self_samples[np.newaxis, :], axis=2
        )
        # Set diagonal to inf so we get nearest-neighbor (not self-distance=0)
        np.fill_diagonal(dists, np.inf)
        nn_dists = np.min(dists, axis=1)  # nearest-neighbor distance per sample

        # r_self = 95th percentile of NN distances
        # This means 95% of self-samples will score 0, eliminating most FPs
        r_self = float(np.percentile(nn_dists, 95))

        # Floor: don't go below 0.05 (degenerate), cap at 2.0
        return max(0.05, min(2.0, r_self))

    def fit(self, self_samples: np.ndarray, max_attempts: int = 5000):
        """
        Training Phase: Thymic Selection.
        Only detectors that don't bind to self (R_q > R_self) become mature.

        If config.adaptive is True, r_self is computed from self-data geometry
        instead of using the fixed config value.
        """
        self.self_samples = np.array(self_samples)
        self.feature_dim = self_samples.shape[1]
        self.valid_detectors = []

        # Adaptive r_self: compute from data geometry
        if self.config.adaptive:
            self._effective_r_self = self._compute_adaptive_r_self(self.self_samples)
        else:
            self._effective_r_self = self.config.r_self

        r_self = self._effective_r_self

        attempts = 0
        while len(self.valid_detectors) < self.config.num_detectors and attempts < max_attempts:
            attempts += 1
            # Generate random candidate in unit hypercube [0, 1]
            candidate_center = np.random.uniform(0, 1, self.feature_dim)

            r_q = self._get_nearest_self_distance(candidate_center)

            # Equation 20: Validation Rule
            if r_q > r_self:
                # Equation 21 Calculation of detector radius
                detector_radius = r_q - r_self

                # Check for duplicates (distinct detector requirement)
                is_duplicate = any(np.allclose(d.center, candidate_center, atol=1e-5) for d in self.valid_detectors)

                if not is_duplicate:
                    self.valid_detectors.append(Detector(
                        center=candidate_center,
                        radius=detector_radius,
                        class_label=self.class_label,
                        r_self=r_self
                    ))

        # Score calibration: compute score distribution on self-data
        # so we can normalize scores to [0, 1] with proper dynamic range
        self._calibrate_scores()

        return self

    def _calibrate_scores(self):
        """Compute score calibration so self→0 and boundary→0.5.

        Strategy: self-samples should score near 0. The max self-sample
        score defines the boundary of "normal." We set normalization so that
        the boundary maps to ~0.3, giving anomalies room to score 0.5–1.0.
        """
        if self.self_samples is None or len(self.self_samples) == 0:
            self._score_min = 0.0
            self._score_max = 1.0
            return

        # Score every self sample
        self_scores = np.array([self._raw_anomaly_score(s) for s in self.self_samples])
        max_self_score = float(np.max(self_scores))

        # The normalizer maps max_self_score → 0.3 (below threshold)
        # and 2 * max_self_score → 0.6 (above threshold for clear anomalies)
        # This ensures the self/anomaly boundary sits near 0.5
        self._score_min = 0.0
        if max_self_score > 0:
            # Normalize so max self score maps to ~0.3
            self._score_max = max_self_score / 0.3
        else:
            # All self-samples score 0 (ideal). Use adaptive r_self as
            # the reference: a point at distance 2*r_self from self
            # (raw score = r_self) should map to ~0.5
            self._score_max = self._effective_r_self / 0.5 if self._effective_r_self > 0 else 1.0

    def predict_single(self, sample: np.ndarray) -> float:
        """
        Classification:
        Eq 20/Section 4.3.4: An anomaly (Non-Self) is detected if:
        1. Low similarity to self (Distance > R_self)
        OR
        2. Binds to a detector (Maturation logic handles this)
        """
        if not self.valid_detectors:
            return 0.0

        r_self = self._effective_r_self
        r_q = self._get_nearest_self_distance(sample)

        # If it's close to self, it's SELF
        if r_q <= r_self:
            return 0.0

        # If it's far from self, check if it binds to our mature detectors
        for d in self.valid_detectors:
            dist = self._euclidean_distance(sample, d.center)
            if dist < d.radius:
                return 1.0

        # Far from self but doesn't bind a detector — still non-self
        return 1.0

    def _raw_anomaly_score(self, sample: np.ndarray) -> float:
        """Raw uncalibrated score: max(0, r_q - r_self)."""
        r_q = self._get_nearest_self_distance(sample)
        return max(0.0, r_q - self._effective_r_self)

    def get_anomaly_score(self, sample: np.ndarray) -> float:
        """Calibrated anomaly score normalized to [0, 1].

        Uses training-time score distribution for proper dynamic range.
        Self-samples score near 0, true outliers score near 1.
        """
        raw = self._raw_anomaly_score(sample)
        # Calibrated normalization against training distribution
        score_range = self._score_max - self._score_min
        if score_range <= 0:
            return 0.0
        normalized = (raw - self._score_min) / score_range
        return float(np.clip(normalized, 0.0, 1.0))

# =============================================================================
# METRICS FOR PREPRINT
# =============================================================================

def calculate_mcc(tp, tn, fp, fn):
    """Matthews Correlation Coefficient (Eq in text)"""
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator > 0 else 0.0

def calculate_kappa(tp, tn, fp, fn):
    """Cohen's Kappa"""
    total = tp + tn + fp + fn
    po = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total**2)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0

def calculate_gen_error(train_err, test_err):
    """Equation 22: E_gen = |(1/N_trg * sum(L)) - (1/N_test * sum(L))|"""
    return abs(train_err - test_err)
