"""
Hallucination-specific Dendritic Agent for IMMUNOS

Extracts features optimized for detecting hallucinated vs truthful text.
Designed for TruthfulQA and similar datasets.
"""

import re
from typing import Dict, Any, List, Set, Optional

import numpy as np

from ..core.antigen import Antigen


class HallucinationDendriticAgent:
    """
    Feature extractor for hallucination detection.

    Extracts domain-specific features:
    - Uncertainty/hedging (truthful answers often hedge)
    - Absoluteness (hallucinations often overstate)
    - Common misconception patterns
    - Grounding/refusal cues
    - Optional: Semantic similarity to truthful prototype
    """

    def __init__(self, agent_name: str = "halluc_dendritic_001"):
        self.agent_name = agent_name
        self._load_patterns()
        self._truthful_prototype: Optional[np.ndarray] = None
        self._embedder = None

    def _load_patterns(self):
        """Load word patterns for hallucination detection."""
        # Uncertainty phrases - truthful answers often use these
        self.uncertainty_phrases: Set[str] = {
            "unclear", "not known", "debated", "varies", "depends",
            "uncertain", "unknown", "disputed", "controversial",
            "no evidence", "no proof", "unproven", "not established",
            "hard to say", "difficult to determine", "not clear",
            "remains unclear", "still debated", "open question",
            "no consensus", "inconclusive", "ambiguous"
        }

        # Hedging words - signal careful/truthful statements
        self.hedging_words: Set[str] = {
            "may", "might", "possibly", "perhaps", "could", "likely",
            "probably", "sometimes", "often", "usually", "generally",
            "tend", "seems", "appears", "suggests", "typically",
            "can", "potentially", "approximately", "roughly", "about",
            "around", "nearly", "almost", "somewhat", "fairly"
        }

        # Absoluteness markers - hallucinations often overstate
        self.absolute_words: Set[str] = {
            "always", "never", "definitely", "certainly", "absolutely",
            "proven", "guaranteed", "undoubtedly", "everyone", "nobody",
            "all", "none", "must", "impossible", "obviously",
            "completely", "totally", "entirely", "perfectly", "exactly",
            "without doubt", "for sure", "100%", "every", "any"
        }

        # Common misconception triggers
        self.misconception_phrases: List[str] = [
            "actually", "contrary to", "myth", "misconception",
            "commonly believed", "in fact", "despite popular belief",
            "widely thought", "people think", "many believe",
            "it's a common", "popular myth", "false belief",
            "not true that", "incorrect to think"
        ]

        # Grounding cues - signal factual basis
        self.grounding_phrases: List[str] = [
            "according to", "research shows", "studies indicate",
            "evidence suggests", "experts say", "scientists found",
            "data shows", "historically", "documented",
            "based on", "as per", "sources indicate",
            "it has been shown", "findings suggest"
        ]

        # Refusal cues - signal honest uncertainty
        self.refusal_phrases: List[str] = [
            "i don't know", "not sure", "can't say", "unclear",
            "no one knows", "impossible to say", "hard to determine",
            "depends on", "varies by", "it's complicated",
            "cannot be determined", "there is no", "we don't know",
            "it depends", "that's uncertain", "no definitive answer"
        ]

        # Confident false claim patterns (hallucination indicators)
        self.overconfident_phrases: List[str] = [
            "is the", "was the", "are the", "were the",
            "will always", "has always been", "is known as",
            "is called", "is named", "is located in"
        ]

        # Specific factual claim patterns
        self.specific_claim_pattern = re.compile(
            r'\b\d+\s*(percent|%|years?|times?|people|countries)\b', re.I
        )

    def extract_features(self, antigen: Antigen) -> Dict[str, Any]:
        """
        Extract hallucination-specific features.

        Returns dict with ~15 features for downstream processing.
        """
        text = antigen.get_text_content()
        text_lower = text.lower()
        words = set(text_lower.split())

        features: Dict[str, Any] = {}

        # Basic structure
        features["length"] = len(text)
        features["tokens"] = len(text.split())

        # Uncertainty features (truthful indicators)
        uncertainty_count = sum(1 for phrase in self.uncertainty_phrases
                               if phrase in text_lower)
        hedging_count = len(words & self.hedging_words)
        features["uncertainty_count"] = uncertainty_count
        features["hedging_count"] = hedging_count
        features["uncertainty_score"] = min((uncertainty_count + hedging_count) / 5.0, 1.0)

        # Absoluteness features (hallucination indicators)
        absolute_count = len(words & self.absolute_words)
        features["absolute_count"] = absolute_count
        features["absoluteness_score"] = min(absolute_count / 3.0, 1.0)

        # Misconception pattern features
        misconception_count = sum(1 for phrase in self.misconception_phrases
                                  if phrase in text_lower)
        features["misconception_count"] = misconception_count
        features["has_misconception_marker"] = misconception_count > 0

        # Grounding features (truthful indicators)
        grounding_count = sum(1 for phrase in self.grounding_phrases
                              if phrase in text_lower)
        features["grounding_count"] = grounding_count
        features["grounding_score"] = min(grounding_count / 2.0, 1.0)

        # Refusal features (truthful indicators)
        refusal_count = sum(1 for phrase in self.refusal_phrases
                            if phrase in text_lower)
        features["refusal_count"] = refusal_count
        features["has_refusal"] = refusal_count > 0

        # Specific claim without grounding (hallucination indicator)
        has_specific_claim = bool(self.specific_claim_pattern.search(text))
        features["has_specific_claim"] = has_specific_claim
        features["ungrounded_claim"] = has_specific_claim and grounding_count == 0

        # Overconfident phrasing (hallucination indicator)
        overconfident_count = sum(1 for phrase in self.overconfident_phrases
                                  if phrase in text_lower)
        features["overconfident_count"] = overconfident_count
        features["overconfident_score"] = min(overconfident_count / 3.0, 1.0)

        # Answer length ratio (short confident answers often hallucinate)
        word_count = len(text.split())
        features["is_short_answer"] = word_count < 15
        features["is_long_answer"] = word_count > 50

        # Composite truthfulness score
        # Higher = more likely truthful
        truthful_signals = (
            features["uncertainty_score"] * 0.25 +
            features["grounding_score"] * 0.25 +
            (0.15 if features["has_refusal"] else 0.0) +
            (0.15 if features["has_misconception_marker"] else 0.0) +
            (0.10 if features["is_long_answer"] else 0.0) +
            (0.10 * features["hedging_count"] / 3.0)
        )
        halluc_signals = (
            features["absoluteness_score"] * 0.3 +
            features["overconfident_score"] * 0.3 +
            (0.2 if features["ungrounded_claim"] else 0.0) +
            (0.2 if features["is_short_answer"] else 0.0)
        )
        features["truthfulness_score"] = max(0.0, min(1.0, 0.5 + truthful_signals - halluc_signals))

        return features

    def set_embedder(self, embedder) -> None:
        """Set the embedder for semantic similarity features."""
        self._embedder = embedder

    def compute_truthful_prototype(self, truthful_texts: List[str]) -> None:
        """
        Compute mean embedding of truthful samples as prototype.

        Args:
            truthful_texts: List of truthful answer texts
        """
        if not self._embedder:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        embeddings = []
        for text in truthful_texts:
            try:
                emb = self._embedder.embed(text)
                embeddings.append(emb)
            except Exception:
                continue

        if embeddings:
            self._truthful_prototype = np.mean(embeddings, axis=0)
            # Normalize for cosine similarity
            norm = np.linalg.norm(self._truthful_prototype)
            if norm > 0:
                self._truthful_prototype = self._truthful_prototype / norm

    def get_prototype_similarity(self, text: str) -> float:
        """
        Compute cosine similarity to truthful prototype.

        Returns 0.5 if no prototype set (neutral).
        """
        if self._truthful_prototype is None or self._embedder is None:
            return 0.5

        try:
            emb = self._embedder.embed(text)
            # Normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            # Cosine similarity (already normalized)
            similarity = float(np.dot(emb, self._truthful_prototype))
            # Scale from [-1, 1] to [0, 1]
            return (similarity + 1.0) / 2.0
        except Exception:
            return 0.5

    def get_feature_vector(self, antigen: Antigen, include_embedding: bool = False) -> List[float]:
        """
        Get numeric feature vector for ML models.

        Args:
            antigen: The antigen to extract features from
            include_embedding: If True, add prototype similarity as 21st feature

        Returns:
            List of 20 float values (or 21 if include_embedding=True)
        """
        features = self.extract_features(antigen)

        vector = [
            # Structure (2)
            float(features.get("length", 0)) / 500.0,
            float(features.get("tokens", 0)) / 50.0,
            # Uncertainty/hedging (3)
            float(features.get("uncertainty_count", 0)) / 5.0,
            float(features.get("hedging_count", 0)) / 5.0,
            float(features.get("uncertainty_score", 0)),
            # Absoluteness (2)
            float(features.get("absolute_count", 0)) / 5.0,
            float(features.get("absoluteness_score", 0)),
            # Misconception (2)
            float(features.get("misconception_count", 0)) / 3.0,
            1.0 if features.get("has_misconception_marker") else 0.0,
            # Grounding (2)
            float(features.get("grounding_count", 0)) / 3.0,
            float(features.get("grounding_score", 0)),
            # Refusal (2)
            float(features.get("refusal_count", 0)) / 3.0,
            1.0 if features.get("has_refusal") else 0.0,
            # Claims (2)
            1.0 if features.get("has_specific_claim") else 0.0,
            1.0 if features.get("ungrounded_claim") else 0.0,
            # Overconfidence (2)
            float(features.get("overconfident_count", 0)) / 5.0,
            float(features.get("overconfident_score", 0)),
            # Answer length (2)
            1.0 if features.get("is_short_answer") else 0.0,
            1.0 if features.get("is_long_answer") else 0.0,
            # Composite (1)
            float(features.get("truthfulness_score", 0.5)),
        ]

        # Optional: Add semantic similarity to truthful prototype (1)
        if include_embedding:
            text = antigen.get_text_content()
            similarity = self.get_prototype_similarity(text)
            vector.append(similarity)

        return vector

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_name": self.agent_name,
            "feature_count": 20,
            "feature_categories": [
                "structure", "uncertainty", "absoluteness",
                "misconception", "grounding", "refusal", "claims",
                "overconfidence", "answer_length", "composite"
            ]
        }
