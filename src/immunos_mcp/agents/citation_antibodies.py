#!/usr/bin/env python3
"""
Citation Antibody System - Multi-Antibody Architecture for Citation Verification
================================================================================
Each citation component (DOI, PMID, title, authors, journal, year) has its own
specialized B Cell antibody trained on real patterns for that specific field.

This reduces false positives by not confusing patterns across different fields.
"""

import re
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from ..algorithms.negsel import NegativeSelectionClassifier, NegSelConfig


@dataclass
class AntibodyResult:
    """Result from a single antibody check."""
    component: str
    is_anomaly: bool
    confidence: float
    matched_pattern: Optional[str] = None
    reason: str = ""


@dataclass
class CitationVerificationResult:
    """Combined result from all antibodies."""
    is_hallucinated: bool
    overall_confidence: float
    component_results: Dict[str, AntibodyResult] = field(default_factory=dict)
    anomaly_count: int = 0
    total_checks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_hallucinated": self.is_hallucinated,
            "overall_confidence": self.overall_confidence,
            "anomaly_count": self.anomaly_count,
            "total_checks": self.total_checks,
            "components": {
                k: {
                    "component": v.component,
                    "is_anomaly": v.is_anomaly,
                    "confidence": v.confidence,
                    "reason": v.reason
                }
                for k, v in self.component_results.items()
            }
        }


class BaseAntibody:
    """Base class for citation component antibodies."""

    def __init__(self, component_name: str, num_detectors: int = 50):
        self.component_name = component_name
        self.patterns: List[str] = []
        self.config = NegSelConfig(
            num_detectors=num_detectors,
            r_self=0.3,
            description=f"{component_name} Antibody"
        )
        self.nk_detector = NegativeSelectionClassifier(config=self.config)
        self.is_trained = False

    def extract_features(self, value: str) -> np.ndarray:
        """Override in subclass to extract component-specific features."""
        raise NotImplementedError

    def train(self, valid_examples: List[str]):
        """Train on valid examples of this component."""
        self.patterns = valid_examples
        if len(valid_examples) >= 3:
            features = np.array([self.extract_features(v) for v in valid_examples])
            self.nk_detector.fit(features)
            self.is_trained = True

    def check(self, value: str) -> AntibodyResult:
        """Check if a value matches known valid patterns."""
        if not value or not value.strip():
            return AntibodyResult(
                component=self.component_name,
                is_anomaly=True,
                confidence=1.0,
                reason="Empty or missing value"
            )

        if not self.is_trained:
            # Fallback to rule-based check
            return self._rule_based_check(value)

        features = self.extract_features(value)
        is_anomaly = self.nk_detector.predict_single(features) == 1.0
        score = self.nk_detector.get_anomaly_score(features)

        return AntibodyResult(
            component=self.component_name,
            is_anomaly=is_anomaly,
            confidence=float(score),
            reason="NK detector flagged as anomaly" if is_anomaly else "Matches self patterns"
        )

    def _rule_based_check(self, value: str) -> AntibodyResult:
        """Fallback rule-based validation when not trained."""
        return AntibodyResult(
            component=self.component_name,
            is_anomaly=False,
            confidence=0.5,
            reason="No training data - using default"
        )

    def save_state(self, path: str):
        """Save antibody state to file."""
        state = {
            "component_name": self.component_name,
            "patterns": self.patterns,
            "is_trained": self.is_trained,
            "config": self.config,
            "nk_detector": self.nk_detector if self.is_trained else None
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load_state(cls, path: str) -> 'BaseAntibody':
        """Load antibody state from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        antibody = cls.__new__(cls)
        antibody.component_name = state["component_name"]
        antibody.patterns = state["patterns"]
        antibody.is_trained = state["is_trained"]
        antibody.config = state["config"]
        antibody.nk_detector = state["nk_detector"] or NegativeSelectionClassifier(config=antibody.config)
        return antibody


class DOIAntibody(BaseAntibody):
    """
    Antibody specialized for DOI (Digital Object Identifier) validation.

    Valid DOI pattern: 10.XXXX/YYYY where XXXX is registrant code
    Examples: 10.1038/nature12373, 10.1016/j.cell.2023.01.001
    """

    # Known DOI registrant prefixes for major publishers
    KNOWN_REGISTRANTS = {
        "1038": "Nature",
        "1016": "Elsevier",
        "1126": "Science",
        "1371": "PLOS",
        "1093": "Oxford",
        "1073": "PNAS",
        "1186": "BMC",
        "1007": "Springer",
        "1002": "Wiley",
        "1101": "Cold Spring Harbor",
        "1161": "AHA",
        "1172": "JCI",
        "7554": "eLife",
    }

    def __init__(self):
        super().__init__("DOI", num_detectors=30)
        self.doi_pattern = re.compile(r'^10\.\d{4,}/[^\s]+$')

    def extract_features(self, doi: str) -> np.ndarray:
        """Extract DOI-specific features."""
        features = []

        # 1. Has valid DOI prefix (10.)
        features.append(1.0 if doi.startswith("10.") else 0.0)

        # 2. Registrant code (4-5 digits after 10.)
        registrant_match = re.match(r'^10\.(\d{4,5})/', doi)
        if registrant_match:
            registrant = registrant_match.group(1)
            features.append(1.0 if registrant in self.KNOWN_REGISTRANTS else 0.5)
            features.append(len(registrant) / 5.0)  # Normalized length
        else:
            features.append(0.0)
            features.append(0.0)

        # 3. Overall DOI length (typical: 20-60 chars)
        features.append(min(len(doi) / 60.0, 1.0))

        # 4. Contains slash separator
        features.append(1.0 if "/" in doi else 0.0)

        # 5. Character entropy (hallucinated DOIs often have unusual entropy)
        entropy = self._calculate_entropy(doi)
        features.append(entropy / 5.0)

        # 6. Number of path segments
        segments = doi.split("/")
        features.append(min(len(segments) / 4.0, 1.0))

        # 7. Contains typical DOI characters only
        valid_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-_/()")
        invalid_ratio = sum(1 for c in doi if c not in valid_chars) / max(len(doi), 1)
        features.append(1.0 - invalid_ratio)

        # Pad to 10 features
        while len(features) < 10:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        total = len(text)
        return -sum((count/total) * np.log2(count/total) for count in freq.values())

    def _rule_based_check(self, doi: str) -> AntibodyResult:
        """Rule-based DOI validation."""
        # Check basic DOI format
        if not self.doi_pattern.match(doi):
            return AntibodyResult(
                component="DOI",
                is_anomaly=True,
                confidence=0.9,
                reason=f"Invalid DOI format: {doi[:30]}..."
            )

        # Check for known registrant
        registrant_match = re.match(r'^10\.(\d{4,5})/', doi)
        if registrant_match:
            registrant = registrant_match.group(1)
            if registrant in self.KNOWN_REGISTRANTS:
                return AntibodyResult(
                    component="DOI",
                    is_anomaly=False,
                    confidence=0.9,
                    matched_pattern=f"Known registrant: {self.KNOWN_REGISTRANTS[registrant]}",
                    reason="Valid DOI format with known publisher"
                )

        return AntibodyResult(
            component="DOI",
            is_anomaly=False,
            confidence=0.6,
            reason="Valid DOI format but unknown registrant"
        )


class PMIDAntibody(BaseAntibody):
    """
    Antibody specialized for PubMed ID (PMID) validation.

    Valid PMID: Numeric, typically 1-8 digits, monotonically increasing over time
    Current range (2024): ~1 to ~39,000,000
    """

    # PMID ranges by approximate year (for temporal validation)
    PMID_RANGES = {
        1970: (1, 100000),
        1980: (100000, 2000000),
        1990: (2000000, 10000000),
        2000: (10000000, 15000000),
        2010: (15000000, 25000000),
        2020: (25000000, 35000000),
        2025: (35000000, 40000000),
    }

    def __init__(self):
        super().__init__("PMID", num_detectors=20)

    def extract_features(self, pmid: str) -> np.ndarray:
        """Extract PMID-specific features."""
        features = []

        # Clean PMID (remove "PMID:" prefix if present)
        pmid_clean = re.sub(r'^PMID:?\s*', '', pmid.strip())

        # 1. Is purely numeric
        is_numeric = pmid_clean.isdigit()
        features.append(1.0 if is_numeric else 0.0)

        if is_numeric:
            pmid_int = int(pmid_clean)

            # 2. Within valid range (1 to ~40 million)
            features.append(1.0 if 1 <= pmid_int <= 40000000 else 0.0)

            # 3. Normalized value (log scale)
            features.append(min(np.log10(pmid_int + 1) / 8.0, 1.0))

            # 4. Length (typically 1-8 digits)
            features.append(min(len(pmid_clean) / 8.0, 1.0))
        else:
            features.extend([0.0, 0.0, 0.0])

        # Pad to 10 features
        while len(features) < 10:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _rule_based_check(self, pmid: str) -> AntibodyResult:
        """Rule-based PMID validation."""
        pmid_clean = re.sub(r'^PMID:?\s*', '', pmid.strip())

        if not pmid_clean.isdigit():
            return AntibodyResult(
                component="PMID",
                is_anomaly=True,
                confidence=0.95,
                reason=f"PMID must be numeric: {pmid}"
            )

        pmid_int = int(pmid_clean)

        if pmid_int < 1 or pmid_int > 40000000:
            return AntibodyResult(
                component="PMID",
                is_anomaly=True,
                confidence=0.9,
                reason=f"PMID out of valid range: {pmid_int}"
            )

        return AntibodyResult(
            component="PMID",
            is_anomaly=False,
            confidence=0.85,
            reason="Valid PMID range"
        )

    def check_temporal_consistency(self, pmid: str, year: int) -> AntibodyResult:
        """Check if PMID is consistent with publication year."""
        pmid_clean = re.sub(r'^PMID:?\s*', '', pmid.strip())

        if not pmid_clean.isdigit():
            return AntibodyResult(
                component="PMID",
                is_anomaly=True,
                confidence=0.9,
                reason="Invalid PMID format for temporal check"
            )

        pmid_int = int(pmid_clean)

        # Find expected range for year
        for ref_year in sorted(self.PMID_RANGES.keys(), reverse=True):
            if year >= ref_year:
                min_pmid, max_pmid = self.PMID_RANGES[ref_year]
                if pmid_int < min_pmid:
                    return AntibodyResult(
                        component="PMID",
                        is_anomaly=True,
                        confidence=0.8,
                        reason=f"PMID {pmid_int} too low for year {year}"
                    )
                break

        return AntibodyResult(
            component="PMID",
            is_anomaly=False,
            confidence=0.7,
            reason="PMID consistent with year"
        )


class TitleAntibody(BaseAntibody):
    """
    Antibody specialized for publication title validation.

    Checks for:
    - Reasonable length
    - Proper capitalization patterns
    - Scientific terminology
    - Absence of hallucination markers (gibberish, repeated words)
    """

    # Common hallucination patterns in titles
    HALLUCINATION_MARKERS = [
        r'\b(\w+)\s+\1\s+\1\b',  # Triple repeated words
        r'[^\x00-\x7F]{5,}',  # Long non-ASCII sequences
        r'^\s*$',  # Empty
        r'^[A-Z\s]+$',  # All caps (unusual for titles)
    ]

    def __init__(self):
        super().__init__("Title", num_detectors=40)
        self.hallucination_patterns = [re.compile(p) for p in self.HALLUCINATION_MARKERS]

    def extract_features(self, title: str) -> np.ndarray:
        """Extract title-specific features."""
        features = []

        # 1. Length (typical: 10-200 chars)
        features.append(min(len(title) / 200.0, 1.0))

        # 2. Word count (typical: 5-25 words)
        words = title.split()
        features.append(min(len(words) / 25.0, 1.0))

        # 3. Average word length
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        features.append(min(avg_word_len / 15.0, 1.0))

        # 4. Capitalization ratio
        upper_count = sum(1 for c in title if c.isupper())
        cap_ratio = upper_count / max(len(title), 1)
        features.append(cap_ratio)

        # 5. Contains colon (common in scientific titles)
        features.append(1.0 if ":" in title else 0.0)

        # 6. Punctuation density
        punct_count = sum(1 for c in title if c in '.,;:!?()-')
        features.append(min(punct_count / max(len(words), 1), 1.0))

        # 7. Has hallucination markers
        has_markers = any(p.search(title) for p in self.hallucination_patterns)
        features.append(1.0 if has_markers else 0.0)

        # 8. Unique word ratio (repeated words = suspicious)
        unique_ratio = len(set(words)) / max(len(words), 1)
        features.append(unique_ratio)

        # Pad to 10 features
        while len(features) < 10:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _rule_based_check(self, title: str) -> AntibodyResult:
        """Rule-based title validation."""
        # Check length
        if len(title) < 10:
            return AntibodyResult(
                component="Title",
                is_anomaly=True,
                confidence=0.85,
                reason="Title too short"
            )

        if len(title) > 500:
            return AntibodyResult(
                component="Title",
                is_anomaly=True,
                confidence=0.8,
                reason="Title unusually long"
            )

        # Check for hallucination markers
        for pattern in self.hallucination_patterns:
            if pattern.search(title):
                return AntibodyResult(
                    component="Title",
                    is_anomaly=True,
                    confidence=0.9,
                    reason="Title contains hallucination markers"
                )

        # Check word repetition
        words = title.lower().split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                return AntibodyResult(
                    component="Title",
                    is_anomaly=True,
                    confidence=0.85,
                    reason="Excessive word repetition in title"
                )

        return AntibodyResult(
            component="Title",
            is_anomaly=False,
            confidence=0.7,
            reason="Title passes basic validation"
        )


class AuthorAntibody(BaseAntibody):
    """
    Antibody specialized for author name validation.

    Checks for:
    - Proper name format (First Last, or F. Last)
    - Reasonable number of authors
    - Absence of gibberish names
    """

    # Suspicious author patterns
    SUSPICIOUS_PATTERNS = [
        r'^[A-Z]{10,}$',  # All caps gibberish
        r'^\d+$',  # Just numbers
        r'^[^a-zA-Z]+$',  # No letters
        r'(.)\1{4,}',  # Repeated characters (aaaa)
    ]

    def __init__(self):
        super().__init__("Authors", num_detectors=30)
        self.suspicious_patterns = [re.compile(p) for p in self.SUSPICIOUS_PATTERNS]

    def extract_features(self, authors: str) -> np.ndarray:
        """Extract author-specific features."""
        features = []

        # Split authors (common separators: , ; and)
        author_list = re.split(r'[,;]|\band\b', authors)
        author_list = [a.strip() for a in author_list if a.strip()]

        # 1. Number of authors (typical: 1-20, some papers have 100+)
        features.append(min(len(author_list) / 20.0, 1.0))

        # 2. Average name length
        avg_len = np.mean([len(a) for a in author_list]) if author_list else 0
        features.append(min(avg_len / 30.0, 1.0))

        # 3. Contains initials (F. or F )
        has_initials = any(re.search(r'\b[A-Z]\.\s', a) for a in author_list)
        features.append(1.0 if has_initials else 0.0)

        # 4. Suspicious pattern ratio
        suspicious_count = sum(
            1 for a in author_list
            if any(p.search(a) for p in self.suspicious_patterns)
        )
        features.append(suspicious_count / max(len(author_list), 1))

        # 5. All authors have at least 2 parts (first + last)
        multi_part = sum(1 for a in author_list if len(a.split()) >= 2)
        features.append(multi_part / max(len(author_list), 1))

        # Pad to 10 features
        while len(features) < 10:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _rule_based_check(self, authors: str) -> AntibodyResult:
        """Rule-based author validation."""
        if not authors or len(authors) < 3:
            return AntibodyResult(
                component="Authors",
                is_anomaly=True,
                confidence=0.9,
                reason="Missing or too short author field"
            )

        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern.search(authors):
                return AntibodyResult(
                    component="Authors",
                    is_anomaly=True,
                    confidence=0.85,
                    reason="Author field contains suspicious pattern"
                )

        return AntibodyResult(
            component="Authors",
            is_anomaly=False,
            confidence=0.7,
            reason="Authors pass basic validation"
        )


class JournalAntibody(BaseAntibody):
    """
    Antibody specialized for journal name validation.

    Can be trained on known journal names for high-confidence matching.
    """

    # High-impact journals (partial list for rule-based fallback)
    KNOWN_JOURNALS = {
        "nature", "science", "cell", "lancet", "nejm", "new england journal of medicine",
        "jama", "bmj", "pnas", "plos one", "plos biology", "elife", "nature medicine",
        "nature genetics", "nature neuroscience", "nature communications", "scientific reports",
        "journal of biological chemistry", "journal of clinical investigation",
        "circulation", "blood", "cancer research", "gastroenterology", "hepatology"
    }

    def __init__(self):
        super().__init__("Journal", num_detectors=40)
        self.known_journals_lower = {j.lower() for j in self.KNOWN_JOURNALS}

    def extract_features(self, journal: str) -> np.ndarray:
        """Extract journal-specific features."""
        features = []
        journal_lower = journal.lower().strip()

        # 1. Length (typical: 5-100 chars)
        features.append(min(len(journal) / 100.0, 1.0))

        # 2. Word count
        words = journal.split()
        features.append(min(len(words) / 10.0, 1.0))

        # 3. Is known journal
        is_known = journal_lower in self.known_journals_lower
        features.append(1.0 if is_known else 0.0)

        # 4. Contains "journal" or "proceedings"
        has_journal_word = "journal" in journal_lower or "proceedings" in journal_lower
        features.append(1.0 if has_journal_word else 0.0)

        # 5. Contains common abbreviations
        has_abbrev = bool(re.search(r'\b(J|Proc|Ann|Int|Am|Br|Eur)\b', journal))
        features.append(1.0 if has_abbrev else 0.0)

        # Pad to 10 features
        while len(features) < 10:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _rule_based_check(self, journal: str) -> AntibodyResult:
        """Rule-based journal validation."""
        journal_lower = journal.lower().strip()

        if len(journal) < 3:
            return AntibodyResult(
                component="Journal",
                is_anomaly=True,
                confidence=0.9,
                reason="Journal name too short"
            )

        # Check if known journal
        if journal_lower in self.known_journals_lower:
            return AntibodyResult(
                component="Journal",
                is_anomaly=False,
                confidence=0.95,
                matched_pattern=journal,
                reason="Known high-impact journal"
            )

        # Check for partial match
        for known in self.known_journals_lower:
            if known in journal_lower or journal_lower in known:
                return AntibodyResult(
                    component="Journal",
                    is_anomaly=False,
                    confidence=0.8,
                    matched_pattern=known,
                    reason="Partial match with known journal"
                )

        return AntibodyResult(
            component="Journal",
            is_anomaly=False,
            confidence=0.5,
            reason="Unknown journal (not necessarily invalid)"
        )


class YearAntibody(BaseAntibody):
    """
    Antibody specialized for publication year validation.
    """

    CURRENT_YEAR = 2026
    MIN_VALID_YEAR = 1800  # First scientific journals

    def __init__(self):
        super().__init__("Year", num_detectors=10)

    def extract_features(self, year: str) -> np.ndarray:
        """Extract year-specific features."""
        features = []

        # Try to parse year
        try:
            year_int = int(str(year).strip())

            # 1. Is in valid range
            is_valid = self.MIN_VALID_YEAR <= year_int <= self.CURRENT_YEAR
            features.append(1.0 if is_valid else 0.0)

            # 2. Normalized year (scaled 1800-2026)
            if is_valid:
                normalized = (year_int - self.MIN_VALID_YEAR) / (self.CURRENT_YEAR - self.MIN_VALID_YEAR)
                features.append(normalized)
            else:
                features.append(0.0)

            # 3. Is recent (last 10 years)
            is_recent = (self.CURRENT_YEAR - 10) <= year_int <= self.CURRENT_YEAR
            features.append(1.0 if is_recent else 0.0)

        except (ValueError, TypeError):
            features.extend([0.0, 0.0, 0.0])

        # Pad to 10 features
        while len(features) < 10:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _rule_based_check(self, year: str) -> AntibodyResult:
        """Rule-based year validation."""
        try:
            year_int = int(str(year).strip())
        except (ValueError, TypeError):
            return AntibodyResult(
                component="Year",
                is_anomaly=True,
                confidence=0.95,
                reason=f"Invalid year format: {year}"
            )

        if year_int > self.CURRENT_YEAR:
            return AntibodyResult(
                component="Year",
                is_anomaly=True,
                confidence=0.95,
                reason=f"Future year: {year_int}"
            )

        if year_int < self.MIN_VALID_YEAR:
            return AntibodyResult(
                component="Year",
                is_anomaly=True,
                confidence=0.9,
                reason=f"Year before scientific publishing: {year_int}"
            )

        return AntibodyResult(
            component="Year",
            is_anomaly=False,
            confidence=0.9,
            reason=f"Valid publication year: {year_int}"
        )


class CitationAntibodySystem:
    """
    Multi-antibody system for comprehensive citation verification.

    Each citation component has its own specialized antibody that can be
    independently trained on real data for that specific field.
    """

    def __init__(self):
        self.antibodies = {
            "doi": DOIAntibody(),
            "pmid": PMIDAntibody(),
            "title": TitleAntibody(),
            "authors": AuthorAntibody(),
            "journal": JournalAntibody(),
            "year": YearAntibody(),
        }

    def train_antibody(self, component: str, valid_examples: List[str]):
        """Train a specific antibody on valid examples."""
        if component not in self.antibodies:
            raise ValueError(f"Unknown component: {component}")
        self.antibodies[component].train(valid_examples)
        print(f"[{component}] Trained on {len(valid_examples)} examples")

    def verify_citation(self, citation: Dict[str, str]) -> CitationVerificationResult:
        """
        Verify a citation using all relevant antibodies.

        Args:
            citation: Dict with keys like 'doi', 'pmid', 'title', 'authors', 'journal', 'year'

        Returns:
            CitationVerificationResult with per-component and overall results
        """
        results = {}
        anomaly_count = 0
        total_checks = 0
        confidence_sum = 0.0

        for component, antibody in self.antibodies.items():
            value = citation.get(component, "")
            if value:  # Only check if component is present
                result = antibody.check(value)
                results[component] = result
                total_checks += 1
                confidence_sum += result.confidence
                if result.is_anomaly:
                    anomaly_count += 1

        # Calculate overall verdict
        if total_checks == 0:
            is_hallucinated = True
            overall_confidence = 1.0
        else:
            # Critical components: DOI, PMID, and year (if obviously invalid)
            critical_anomaly = False
            for critical in ['doi', 'pmid', 'year']:
                if critical in results and results[critical].is_anomaly:
                    critical_anomaly = True
                    break

            # Hallucinated if:
            # - Any critical identifier (DOI/PMID/Year) is anomalous, OR
            # - Any component is flagged with high confidence (>0.5), OR
            # - 2+ components are anomalous
            high_confidence_anomaly = any(
                r.is_anomaly and r.confidence > 0.5
                for r in results.values()
            )
            is_hallucinated = critical_anomaly or high_confidence_anomaly or anomaly_count >= 2
            overall_confidence = confidence_sum / total_checks

        return CitationVerificationResult(
            is_hallucinated=is_hallucinated,
            overall_confidence=overall_confidence,
            component_results=results,
            anomaly_count=anomaly_count,
            total_checks=total_checks
        )

    def save_all(self, directory: str):
        """Save all antibodies to a directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        for name, antibody in self.antibodies.items():
            antibody.save_state(str(path / f"{name}_antibody.pkl"))
        print(f"Saved all antibodies to {directory}")

    def load_all(self, directory: str):
        """Load all antibodies from a directory."""
        path = Path(directory)

        for name in self.antibodies.keys():
            antibody_path = path / f"{name}_antibody.pkl"
            if antibody_path.exists():
                # Load the antibody state
                with open(antibody_path, 'rb') as f:
                    state = pickle.load(f)
                self.antibodies[name].patterns = state.get("patterns", [])
                self.antibodies[name].is_trained = state.get("is_trained", False)
                if state.get("nk_detector"):
                    self.antibodies[name].nk_detector = state["nk_detector"]
        print(f"Loaded antibodies from {directory}")

    def get_training_status(self) -> Dict[str, bool]:
        """Get training status of all antibodies."""
        return {name: ab.is_trained for name, ab in self.antibodies.items()}


# Convenience function
def create_citation_antibody_system() -> CitationAntibodySystem:
    """Create a new citation antibody system."""
    return CitationAntibodySystem()
