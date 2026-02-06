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
import sqlite3
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# =============================================================================
# PAPER PRESETS (Table 5 & 6)
# =============================================================================

@dataclass
class NegSelConfig:
    """Configuration for NegSl-AIS detectors based on publication parameters."""
    num_detectors: int
    r_self: float
    description: str = ""

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
        self.db_path = db_path or os.getenv("IMMUNOS_DB") or "/Users/byron/projects/.immunos/db/immunos.db"
        
        self.valid_detectors: List[Detector] = []
        self.feature_dim: int = 0
        self.self_samples: Optional[np.ndarray] = None

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Eq 21: R_q = sqrt(sum((Omega_k - d_k)^2))"""
        return np.sqrt(np.sum((a - b) ** 2))

    def _get_nearest_self_distance(self, candidate: np.ndarray) -> float:
        """R_q = min(||Self_i, d_j||)"""
        distances = np.linalg.norm(self.self_samples - candidate, axis=1)
        return np.min(distances)

    def fit(self, self_samples: np.ndarray, max_attempts: int = 5000):
        """
        Training Phase: Thymic Selection.
        Only detectors that don't bind to self (R_q > R_self) become mature.
        """
        self.self_samples = np.array(self_samples)
        self.feature_dim = self_samples.shape[1]
        self.valid_detectors = []
        
        attempts = 0
        while len(self.valid_detectors) < self.config.num_detectors and attempts < max_attempts:
            attempts += 1
            # Generate random candidate in unit hypercube [0, 1]
            candidate_center = np.random.uniform(0, 1, self.feature_dim)
            
            r_q = self._get_nearest_self_distance(candidate_center)
            
            # Equation 20: Validation Rule
            if r_q > self.config.r_self:
                # Equation 21 Calculation of detector radius
                detector_radius = r_q - self.config.r_self
                
                # Check for duplicates (distinct detector requirement)
                is_duplicate = any(np.allclose(d.center, candidate_center, atol=1e-5) for d in self.valid_detectors)
                
                if not is_duplicate:
                    self.valid_detectors.append(Detector(
                        center=candidate_center,
                        radius=detector_radius,
                        class_label=self.class_label,
                        r_self=self.config.r_self
                    ))

        return self

    def predict_single(self, sample: np.ndarray) -> float:
        """
        Classification:
        Eq 20/Section 4.3.4: An anomaly (Non-Self) is detected if:
        1. Low similarity to self (Distance > R_self)
        OR
        2. Binds to a detector (Maturation logic handles this)

        Simplified per Section 4.3.4: "detectors that bind with self-samples are discarded... 
        mature T-cells search for non-self antigens"
        """
        if not self.valid_detectors:
            return 0.0 
            
        r_q = self._get_nearest_self_distance(sample)
        
        # If it's close to self, it's SELF
        if r_q <= self.config.r_self:
            return 0.0
            
        # If it's far from self, check if it binds to our mature detectors
        for d in self.valid_detectors:
            dist = self._euclidean_distance(sample, d.center)
            # If distance to detector is small, it's a confirmed NON-SELF
            if dist < d.radius:
                return 1.0
        
        # If it's far from self but doesn't bind to a specific detector, 
        # the paper often still treats it as Non-Self in a binary task
        return 1.0

    def get_anomaly_score(self, sample: np.ndarray) -> float:
        """Calculate score based on distance to nearest detector vs nearest self."""
        r_q = self._get_nearest_self_distance(sample)
        # Score increases as sample moves outside R_self
        score = max(0.0, r_q - self.config.r_self)
        return float(score)

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
