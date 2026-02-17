#!/usr/bin/env python3
"""
Citation Anomaly Detector (NegSl-AIS Applied to LLM Citations)
=============================================================
Detects potential hallucinated citations by scanning for "Non-Self"
metadata patterns (Outside Black Box) and consistency gaps (Inside Black Box).
"""

import numpy as np
from typing import Dict, List, Tuple
from ..algorithms.negsel import NegativeSelectionClassifier, NegSelConfig

class CitationAnomalyDetector:
    def __init__(self):
        # Configuration optimal for metadata anomaly detection
        self.config = NegSelConfig(num_detectors=30, r_self=0.25, description="Citation Guardian")
        self.clf = NegativeSelectionClassifier(config=self.config)

    def extract_metadata_features(self, citation: Dict[str, str]) -> np.ndarray:
        """
        Extracts features from a citation dictionary.
        Keys: 'title', 'authors', 'journal', 'doi', 'year'
        """
        features = []

        # 1. Title/Abstract length ratio (Placeholder)
        features.append(len(citation.get('title', '')) / 100.0)

        # 2. DOI Complexity (High entropy in hallucinated DOIs)
        doi = citation.get('doi', '')
        entropy = self._calculate_entropy(doi)
        features.append(entropy / 8.0) # Normalized

        # 3. Year plausibility
        year = int(citation.get('year', 0))
        features.append(1.0 if (2025 >= year >= 1900) else 0.0)

        # 4. Padding to reach representative feature dim
        features.extend([0.0] * 17)

        return np.array(features, dtype=np.float32)

    def _calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        total = len(text)
        return -sum((count/total) * np.log2(count/total) for count in freq.values())

    def train_on_verified(self, verified_citations: List[Dict[str, str]]):
        """Build the 'Self' model using valid citation samples."""
        self_features = np.array([self.extract_metadata_features(c) for c in verified_citations])
        self.clf.fit(self_features)

    def predict(self, citation: Dict[str, str]) -> Tuple[bool, float]:
        """Returns (is_hallucinated, confidence)"""
        feat = self.extract_metadata_features(citation)
        is_non_self = self.clf.predict_single(feat)
        score = self.clf.get_anomaly_score(feat)

        return (is_non_self == 1.0, score)
