"""
Dendritic Agent for IMMUNOS

Extracts features and signals from antigens for downstream agents.
Implements PAMP (Pathogen-Associated Molecular Patterns), danger signals,
and safe signals for immune response coordination.
"""

import re
from typing import Dict, Any, List, Set
from ..core.antigen import Antigen


class DendriticAgent:
    """
    Feature extractor and signal aggregator.

    Extracts 20+ features for downstream B Cell and NK Cell processing:
    - Text structure (5 features)
    - Claim characteristics (6 features)
    - Semantic signals (5 features)
    - Danger signals (4 features)
    """

    def __init__(self, agent_name: str = "dendritic_001"):
        self.agent_name = agent_name
        self._load_patterns()

    def _load_patterns(self):
        """Load word patterns for feature extraction."""
        self.hedging_words: Set[str] = {
            "may", "might", "possibly", "perhaps", "could", "likely",
            "suggest", "appears", "seems", "indicate", "potentially"
        }
        self.certainty_words: Set[str] = {
            "always", "never", "proven", "definitely", "certainly",
            "undoubtedly", "clearly", "obviously", "absolutely", "must"
        }
        self.exaggeration_words: Set[str] = {
            "revolutionary", "breakthrough", "unprecedented", "miraculous",
            "amazing", "incredible", "extraordinary", "groundbreaking"
        }
        self.negation_words: Set[str] = {
            "not", "no", "never", "neither", "nor", "none", "nothing",
            "nobody", "nowhere", "hardly", "barely", "scarcely"
        }
        self.citation_patterns: List[re.Pattern] = [
            re.compile(r'\b10\.\d{4,}/\S+'),  # DOI
            re.compile(r'PMID:\s*\d+'),       # PubMed ID
            re.compile(r'\[\d+\]'),           # Numbered citation
            re.compile(r'\(\w+\s+et\s+al\.,?\s+\d{4}\)'),  # Author et al. (year)
            re.compile(r'\(\w+,?\s+\d{4}\)')  # (Author, year)
        ]
        self.danger_patterns: List[re.Pattern] = [
            re.compile(r'\bcure[sd]?\b', re.I),
            re.compile(r'\b100\s*%\b'),
            re.compile(r'\bguarantee[sd]?\b', re.I),
            re.compile(r'\bproven\s+to\b', re.I),
            re.compile(r'\bno\s+side\s+effects?\b', re.I)
        ]

    def extract_features(self, antigen: Antigen) -> Dict[str, Any]:
        """
        Extract comprehensive feature vector from antigen.

        Returns dict with 20+ features for downstream processing.
        """
        text = antigen.get_text_content()
        features: Dict[str, Any] = {}

        # Text structure (5 features)
        features.update(self._text_structure(text))

        # Claim characteristics (6 features)
        features.update(self._claim_characteristics(text))

        # Semantic signals (5 features)
        features.update(self._semantic_signals(text))

        # Danger signals (4 features)
        features.update(self._danger_signals(text))

        return features

    def _text_structure(self, text: str) -> Dict[str, Any]:
        """Extract text structure features (5 features)."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        word_lengths = [len(w) for w in words] if words else [0]
        sentence_lengths = [len(s.split()) for s in sentences] if sentences else [0]

        return {
            "length": len(text),
            "tokens": len(words),
            "sentences": len(sentences),
            "avg_word_length": sum(word_lengths) / len(word_lengths) if word_lengths else 0,
            "avg_sentence_length": sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        }

    def _claim_characteristics(self, text: str) -> Dict[str, Any]:
        """Extract claim-specific features (6 features)."""
        text_lower = text.lower()
        words = set(text_lower.split())

        # Check for citations
        has_citation = any(p.search(text) for p in self.citation_patterns)

        # Check for numbers/statistics
        has_numbers = bool(re.search(r'\b\d+\.?\d*\s*%|\b\d+\.?\d*\s*(mg|kg|ml|mmol|patients?|subjects?|years?)\b', text_lower))

        # Hedging and certainty
        hedging_count = len(words & self.hedging_words)
        certainty_count = len(words & self.certainty_words)

        # Questions and negations
        question_count = text.count('?')
        negation_count = len(words & self.negation_words)

        return {
            "has_citation": has_citation,
            "has_numbers": has_numbers,
            "has_hedging": hedging_count > 0,
            "hedging_count": hedging_count,
            "has_certainty": certainty_count > 0,
            "certainty_count": certainty_count,
            "question_count": question_count,
            "negation_count": negation_count
        }

    def _semantic_signals(self, text: str) -> Dict[str, Any]:
        """Extract semantic signal features (5 features)."""
        text_lower = text.lower()
        words = text_lower.split()
        word_set = set(words)

        # Exaggeration score (superlatives and hyperbole)
        exaggeration_count = len(word_set & self.exaggeration_words)
        exaggeration_score = min(exaggeration_count / 3.0, 1.0)  # Normalize

        # Specificity (named entities heuristic - capitalized words)
        capitalized = [w for w in text.split() if w and w[0].isupper() and len(w) > 1]
        specificity_score = min(len(capitalized) / max(len(words), 1) * 5, 1.0)

        # Subjectivity heuristic (first-person pronouns, opinion words)
        subjective_words = {"i", "we", "my", "our", "believe", "think", "feel", "opinion"}
        subjective_count = len(word_set & subjective_words)
        subjectivity_score = min(subjective_count / 5.0, 1.0)

        # Simple sentiment (positive vs negative word ratio)
        positive_words = {"good", "better", "best", "effective", "success", "improve", "benefit"}
        negative_words = {"bad", "worse", "worst", "fail", "harm", "risk", "danger", "death"}
        pos_count = len(word_set & positive_words)
        neg_count = len(word_set & negative_words)
        total = pos_count + neg_count
        sentiment_score = (pos_count - neg_count) / total if total > 0 else 0.0

        # Controversy score (debate markers)
        controversy_words = {"controversial", "debate", "disputed", "disagree", "conflict"}
        controversy_count = len(word_set & controversy_words)
        controversy_score = min(controversy_count / 2.0, 1.0)

        return {
            "exaggeration_score": exaggeration_score,
            "specificity_score": specificity_score,
            "subjectivity_score": subjectivity_score,
            "sentiment_score": sentiment_score,
            "controversy_score": controversy_score
        }

    def _danger_signals(self, text: str) -> Dict[str, Any]:
        """
        Extract danger signal features (4 features).

        PAMP-like patterns indicate potential threats/misinformation.
        """
        # PAMP score (known threat patterns)
        pamp_matches = sum(1 for p in self.danger_patterns if p.search(text))
        pamp_score = min(pamp_matches / 3.0, 1.0)

        # Danger signal count (red flags)
        danger_count = pamp_matches

        # Contradiction indicator (contains both positive and negative claims)
        has_positive = bool(re.search(r'\b(effective|works|proven|successful)\b', text, re.I))
        has_negative = bool(re.search(r'\b(ineffective|fails?|disproven|unsuccessful)\b', text, re.I))
        has_contradiction = has_positive and has_negative

        # Source credibility heuristic
        # Higher if has citations, uses hedging, avoids exaggeration
        features = self._claim_characteristics(text)
        semantic = self._semantic_signals(text)
        credibility = 0.5  # Base
        if features.get("has_citation"):
            credibility += 0.2
        if features.get("has_hedging"):
            credibility += 0.1
        if semantic.get("exaggeration_score", 0) > 0.5:
            credibility -= 0.2
        if pamp_score > 0.5:
            credibility -= 0.3
        credibility = max(0.0, min(1.0, credibility))

        return {
            "pamp_score": pamp_score,
            "danger_signal_count": danger_count,
            "has_contradiction": has_contradiction,
            "source_credibility": credibility
        }

    def classify_signals(self, features: Dict[str, Any]) -> Dict[str, str]:
        """
        Classify overall signal type based on features.

        Returns signal classification for immune coordination.
        """
        pamp = features.get("pamp_score", 0)
        danger = features.get("danger_signal_count", 0)
        credibility = features.get("source_credibility", 0.5)

        if pamp > 0.5 or danger >= 2:
            signal_type = "DANGER"
        elif credibility > 0.7 and pamp == 0:
            signal_type = "SAFE"
        else:
            signal_type = "NEUTRAL"

        return {
            "signal_type": signal_type,
            "confidence": max(pamp, 1 - credibility, credibility)
        }

    def get_feature_vector(self, antigen: Antigen) -> List[float]:
        """
        Get numeric feature vector for ML models.

        Returns list of 20 float values suitable for scikit-learn, etc.
        """
        features = self.extract_features(antigen)

        # Convert to numeric vector (order matters for consistency)
        vector = [
            # Text structure (5)
            float(features.get("length", 0)) / 1000.0,  # Normalize
            float(features.get("tokens", 0)) / 100.0,
            float(features.get("sentences", 0)) / 10.0,
            float(features.get("avg_word_length", 0)) / 10.0,
            float(features.get("avg_sentence_length", 0)) / 20.0,
            # Claim characteristics (6)
            1.0 if features.get("has_citation") else 0.0,
            1.0 if features.get("has_numbers") else 0.0,
            1.0 if features.get("has_hedging") else 0.0,
            1.0 if features.get("has_certainty") else 0.0,
            float(features.get("question_count", 0)) / 5.0,
            float(features.get("negation_count", 0)) / 5.0,
            # Semantic signals (5)
            float(features.get("exaggeration_score", 0)),
            float(features.get("specificity_score", 0)),
            float(features.get("subjectivity_score", 0)),
            float(features.get("sentiment_score", 0)),
            float(features.get("controversy_score", 0)),
            # Danger signals (4)
            float(features.get("pamp_score", 0)),
            float(features.get("danger_signal_count", 0)) / 5.0,
            1.0 if features.get("has_contradiction") else 0.0,
            float(features.get("source_credibility", 0.5)),
        ]

        return vector

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_name": self.agent_name,
            "feature_count": 20,
            "feature_categories": ["text_structure", "claim_characteristics",
                                   "semantic_signals", "danger_signals"]
        }
