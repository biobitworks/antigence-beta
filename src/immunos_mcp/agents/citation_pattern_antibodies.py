#!/usr/bin/env python3
"""
Citation Pattern Antibody System - Multi-Antibody Architecture for Citation Pattern Analysis
=============================================================================================
Complements the existing CitationAntibodySystem (which validates individual fields like
DOI, PMID, title) by analyzing higher-level citation patterns: retracted papers,
self-citation clusters, and predatory journals.

Input: Citation metadata or reference list text (str)
"""

import re
import sys
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from ..algorithms.negsel import NegativeSelectionClassifier, NegSelConfig
from ..core.fusion import ImmuneSignalFusion
from ..core.immune_response import ImmuneResponse


@dataclass
class CitationPatternAntibodyResult:
    """Result from a single citation pattern antibody check."""
    component: str
    is_anomaly: bool
    confidence: float
    matched_pattern: Optional[str] = None
    reason: str = ""
    binding_affinity: float = 0.0


@dataclass
class CitationPatternResult:
    """Combined result from all citation pattern antibodies."""
    is_suspicious: bool
    overall_confidence: float
    response: ImmuneResponse = ImmuneResponse.IGNORE
    component_results: Dict[str, CitationPatternAntibodyResult] = field(default_factory=dict)
    anomaly_count: int = 0
    total_checks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_suspicious": self.is_suspicious,
            "overall_confidence": self.overall_confidence,
            "response": self.response.value,
            "anomaly_count": self.anomaly_count,
            "total_checks": self.total_checks,
            "components": {
                k: {
                    "component": v.component,
                    "is_anomaly": v.is_anomaly,
                    "confidence": v.confidence,
                    "reason": v.reason,
                    "binding_affinity": v.binding_affinity,
                }
                for k, v in self.component_results.items()
            },
        }


class BaseCitationPatternAntibody:
    """Base class for citation pattern antibodies."""

    def __init__(self, component_name: str, num_detectors: int = 50):
        self.component_name = component_name
        self.patterns: List[str] = []
        self.config = NegSelConfig(
            num_detectors=num_detectors,
            r_self=0.85,
            description=f"{component_name} Citation Pattern Antibody",
            adaptive=True,
        )
        self.nk_detector = NegativeSelectionClassifier(config=self.config)
        self.is_trained = False

    def extract_features(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def train(self, valid_examples: List[str]):
        self.patterns = valid_examples
        if len(valid_examples) >= 3:
            features = np.array([self.extract_features(v) for v in valid_examples])
            self.nk_detector.fit(features)
            self.is_trained = True

    def check(self, text: str) -> CitationPatternAntibodyResult:
        if not text or not text.strip():
            return CitationPatternAntibodyResult(
                component=self.component_name,
                is_anomaly=True, confidence=1.0, binding_affinity=1.0,
                reason="Empty or missing text",
            )

        features = self.extract_features(text)

        if not self.is_trained:
            binding = self._bootstrap_binding(features)
        else:
            binding = self.nk_detector.get_anomaly_score(features)

        normalized = min(1.0, max(0.0, binding))

        rule_result = self._rule_based_check(text)
        if rule_result.confidence >= 0.75:
            is_anomaly = rule_result.is_anomaly
        else:
            is_anomaly = normalized > 0.3

        return CitationPatternAntibodyResult(
            component=self.component_name,
            is_anomaly=is_anomaly,
            confidence=rule_result.confidence,
            binding_affinity=normalized,
            matched_pattern=rule_result.matched_pattern,
            reason=rule_result.reason,
        )

    def _bootstrap_binding(self, features: np.ndarray) -> float:
        self_examples = self._generate_self_examples()
        if len(self_examples) >= 3:
            self_features = np.array(
                [self.extract_features(v) for v in self_examples], dtype=np.float32,
            )
            self.nk_detector.fit(self_features)
            self.is_trained = True
            return self.nk_detector.get_anomaly_score(features)
        return 0.5

    def _generate_self_examples(self) -> List[str]:
        return []

    def _rule_based_check(self, text: str) -> CitationPatternAntibodyResult:
        return CitationPatternAntibodyResult(
            component=self.component_name,
            is_anomaly=False, confidence=0.5,
            reason="No training data - using default",
        )

    def save_state(self, path: str):
        state = {
            "component_name": self.component_name,
            "patterns": self.patterns,
            "is_trained": self.is_trained,
            "config": self.config,
            "nk_detector": self.nk_detector if self.is_trained else None,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_state(cls, path: str) -> "BaseCitationPatternAntibody":
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError, EOFError) as e:
            raise RuntimeError(f"Failed to load antibody state from {path}: {e}") from e
        antibody = cls.__new__(cls)
        antibody.component_name = state["component_name"]
        antibody.patterns = state["patterns"]
        antibody.is_trained = state["is_trained"]
        antibody.config = state["config"]
        antibody.nk_detector = state["nk_detector"] or NegativeSelectionClassifier(
            config=antibody.config
        )
        return antibody


# ---------------------------------------------------------------------------
# Antibody 1: Retraction Antibody
# ---------------------------------------------------------------------------

class RetractionAntibody(BaseCitationPatternAntibody):
    """
    Detects retracted or problematic citations.

    Red flags: Known retracted DOI/PMID, "expression of concern", retraction notice
    Quality signals: DOI resolves, reputable journal, recent publication
    """

    # Retraction Watch markers (text patterns, not full database)
    RETRACTION_MARKERS = [
        "retracted", "retraction", "withdrawn", "expression of concern",
        "correction notice", "erratum", "editorial concern",
        "data fabrication", "data falsification", "misconduct",
    ]

    def __init__(self):
        super().__init__("Retraction", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "Smith J et al. (2020) Nature 580:123-130. DOI: 10.1038/s41586-020-1234-5. PMID: 32123456.",
            "Zhang Y, Wang X (2021) Cell Reports 35:109123. DOI: 10.1016/j.celrep.2021.109123. PubMed indexed.",
            "Kumar A et al. (2019) PNAS 116:12345-12350. DOI: 10.1073/pnas.1912345116. Peer-reviewed.",
            "Johnson M, Williams R (2022) Science 375:eabl1234. DOI: 10.1126/science.abl1234. No corrections.",
            "Lee S et al. (2023) Nature Medicine 29:100-110. DOI: 10.1038/s41591-023-1234-5. Current as of 2024.",
            "Anderson P, Thomas B (2020) Lancet 395:1000-1008. DOI: 10.1016/S0140-6736(20)12345-6.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        # RED FLAGS
        # 1. Retraction/withdrawal markers (excluding negated forms) — binary
        has_retraction = any(marker in text_lower for marker in self.RETRACTION_MARKERS)
        is_negated = bool(re.search(r'(?:no|not|without|zero|never)\s+(?:retract|withdraw|concern|correct|erratum|fabricat|falsif|misconduct)', text_lower))
        has_retraction = has_retraction and not is_negated
        features.append(1.0 if has_retraction else 0.0)

        # 2. "Expression of concern" — binary
        has_eoc = bool(re.search(r'expression\s*of\s*concern|editorial\s*concern', text_lower))
        features.append(1.0 if has_eoc else 0.0)

        # 3. Known problematic patterns (fabrication, falsification) — binary
        has_misconduct = bool(re.search(r'fabricat|falsif|misconduct|fraud|plagiari', text_lower))
        features.append(1.0 if has_misconduct else 0.0)

        # 4. Correction/erratum without original being valid — binary
        has_correction = bool(re.search(r'correction|erratum|corrigendum', text_lower))
        has_valid_ref = bool(re.search(r'(?:correction|erratum)\s*(?:to|for|of)', text_lower))
        features.append(1.0 if has_correction and not has_valid_ref else 0.0)

        # 5. Very old DOI with no recent verification — binary
        years = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', text)
        if years:
            oldest = min(int(y) for y in years)
            features.append(1.0 if oldest < 2000 and not bool(re.search(r'verified|confirmed|current|valid', text_lower)) else 0.0)
        else:
            features.append(0.0)

        # QUALITY SIGNALS
        # 6. DOI present — binary
        has_doi = bool(re.search(r'10\.\d{4,}/', text))
        features.append(1.0 if has_doi else 0.0)

        # 7. PMID present — binary
        has_pmid = bool(re.search(r'pmid|pubmed', text_lower))
        features.append(1.0 if has_pmid else 0.0)

        # 8. Reputable journal — binary
        reputable = bool(re.search(r'nature|science|cell|lancet|nejm|jama|bmj|pnas|plos|elife', text_lower))
        features.append(1.0 if reputable else 0.0)

        # 9. Recent publication (2020+) — binary
        if years:
            newest = max(int(y) for y in years)
            features.append(1.0 if newest >= 2020 else 0.0)
        else:
            features.append(0.0)

        # 10. Peer-reviewed indication — binary
        has_peer_review = bool(re.search(r'peer[\s-]?review|indexed|impact\s*factor|scopus|web\s*of\s*science', text_lower))
        features.append(1.0 if has_peer_review else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> CitationPatternAntibodyResult:
        text_lower = text.lower()

        # Check for retraction markers but exclude negated forms ("no retraction", "not retracted")
        has_retraction = any(marker in text_lower for marker in ["retracted", "retraction", "withdrawn"])
        is_negated = bool(re.search(r'(?:no|not|without|zero|never)\s+(?:retract|withdraw)', text_lower))

        if has_retraction and not is_negated:
            return CitationPatternAntibodyResult(
                component="Retraction", is_anomaly=True, confidence=0.95,
                reason="Citation contains retraction/withdrawal markers",
            )

        if "expression of concern" in text_lower:
            return CitationPatternAntibodyResult(
                component="Retraction", is_anomaly=True, confidence=0.85,
                reason="Citation has expression of concern",
            )

        return CitationPatternAntibodyResult(
            component="Retraction", is_anomaly=False, confidence=0.7,
            reason="No retraction markers detected",
        )


# ---------------------------------------------------------------------------
# Antibody 2: Self-Citation Antibody
# ---------------------------------------------------------------------------

class SelfCitationAntibody(BaseCitationPatternAntibody):
    """
    Detects excessive self-citation: >30% self-citation rate, circular clusters,
    own preprints only.

    Red flags: >30% self-citation, circular citation cluster, only own preprints
    Quality signals: Diverse sources, international, multi-decade span
    """

    def __init__(self):
        super().__init__("SelfCitation", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "References: Smith 2020, Jones 2019, Zhang 2021, Kumar 2018, Lee 2022, Anderson 2020, Brown 2017, Garcia 2019, Muller 2021, Johnson 2023.",
            "1. Nature 2020. 2. Science 2019. 3. Cell 2021. 4. PNAS 2020. 5. Lancet 2022. 6. BMJ 2019. 7. JAMA 2021. 8. NEJM 2020.",
            "Diverse citation sources spanning 2005-2023, from 15 different research groups across 8 countries.",
            "References include seminal works (Watson 1953, Crick 1962) and recent findings (2020-2024) from multiple independent laboratories.",
            "Bibliography: 45 references from 30 unique author groups. Self-citation rate: 8% (4/45). International: USA, UK, Japan, Germany, China.",
            "Citation diversity: 38 unique journals, 12 countries, publication years 1998-2024. No single author group >10% of references.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        # Extract author names (simplified: capitalized words before years)
        author_refs = re.findall(r'([A-Z][a-z]+)\s*(?:et\s*al\.?)?\s*(?:\(?\d{4}\)?)', text)
        unique_authors = set(a.lower() for a in author_refs)

        # Count total references
        ref_count = len(re.findall(r'\b(?:19|20)\d{2}\b', text))

        # RED FLAGS
        # 1. Single author dominates (>30% of references) — binary
        if author_refs and ref_count >= 5:
            from collections import Counter
            counts = Counter(a.lower() for a in author_refs)
            max_ratio = max(counts.values()) / ref_count
            features.append(1.0 if max_ratio > 0.3 else 0.0)
        else:
            features.append(0.0)

        # 2. Very few unique authors for reference count — binary
        if ref_count >= 5:
            author_diversity = len(unique_authors) / max(ref_count, 1)
            features.append(1.0 if author_diversity < 0.3 else 0.0)
        else:
            features.append(0.0)

        # 3. Only preprints/non-peer-reviewed — binary
        has_preprint_only = bool(re.search(r'(?:preprint|arxiv|biorxiv|medrxiv)', text_lower))
        has_journal = bool(re.search(r'nature|science|cell|lancet|journal|proceedings|plos|review', text_lower))
        features.append(1.0 if has_preprint_only and not has_journal and ref_count >= 3 else 0.0)

        # 4. Circular citation pattern (same authors citing each other) — binary
        features.append(1.0 if len(unique_authors) <= 2 and ref_count >= 5 else 0.0)

        # 5. Narrow year range (all within 2 years) — binary
        years = [int(y) for y in re.findall(r'\b((?:19|20)\d{2})\b', text)]
        if len(years) >= 3:
            year_span = max(years) - min(years)
            features.append(1.0 if year_span <= 2 else 0.0)
        else:
            features.append(0.0)

        # QUALITY SIGNALS
        # 6. Diverse author set (>5 unique for >10 refs) — binary
        features.append(1.0 if len(unique_authors) >= 5 and ref_count >= 10 else 0.0)

        # 7. Multi-decade span — binary
        if len(years) >= 3:
            features.append(1.0 if max(years) - min(years) >= 10 else 0.0)
        else:
            features.append(0.0)

        # 8. International/diverse journal set — binary
        journal_count = len(re.findall(r'(?:nature|science|cell|lancet|pnas|plos|bmj|jama|nejm|elife)', text_lower))
        features.append(1.0 if journal_count >= 3 else 0.0)

        # 9. Self-citation rate explicitly reported — binary
        has_rate = bool(re.search(r'self[\s-]?citation\s*(?:rate|ratio|percentage)', text_lower))
        features.append(1.0 if has_rate else 0.0)

        # 10. Adequate reference count (>=10) — binary
        features.append(1.0 if ref_count >= 10 else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> CitationPatternAntibodyResult:
        author_refs = re.findall(r'([A-Z][a-z]+)\s*(?:et\s*al\.?)?\s*(?:\(?\d{4}\)?)', text)
        ref_count = len(re.findall(r'\b(?:19|20)\d{2}\b', text))

        if author_refs and ref_count >= 5:
            from collections import Counter
            counts = Counter(a.lower() for a in author_refs)
            max_author, max_count = counts.most_common(1)[0]
            max_ratio = max_count / ref_count
            if max_ratio > 0.3:
                return CitationPatternAntibodyResult(
                    component="SelfCitation", is_anomaly=True, confidence=0.8,
                    reason=f"Excessive self-citation: '{max_author}' appears in {max_ratio:.0%} of references",
                )

        return CitationPatternAntibodyResult(
            component="SelfCitation", is_anomaly=False, confidence=0.6,
            reason="Citation diversity appears acceptable",
        )


# ---------------------------------------------------------------------------
# Antibody 3: Predatory Journal Antibody
# ---------------------------------------------------------------------------

class PredatoryJournalAntibody(BaseCitationPatternAntibody):
    """
    Detects predatory journal markers: Beall's list patterns, not indexed,
    unrealistic review timelines.

    Red flags: Beall's list patterns, not in DOAJ/Scopus, fast review
    Quality signals: PubMed indexed, Clarivate IF, COPE member
    """

    # Common predatory journal name patterns
    PREDATORY_PATTERNS = [
        r'international\s*journal\s*of\s*(?:advanced|emerging|innovative|modern|novel)',
        r'global\s*journal\s*of',
        r'(?:american|european|asian)\s*journal\s*of\s*(?:scientific|academic)',
        r'journal\s*of\s*(?:advanced|emerging)\s*research',
        r'open\s*access\s*(?:journal|publisher)',
        r'rapid\s*publication',
        r'(?:submit|accepted|published)\s*(?:within|in)\s*(?:\d+\s*)?(?:day|hour|week)',
    ]

    def __init__(self):
        super().__init__("PredatoryJournal", num_detectors=30)
        self.predatory_re = [re.compile(p, re.I) for p in self.PREDATORY_PATTERNS]

    def _generate_self_examples(self) -> List[str]:
        return [
            "Published in Nature (Impact Factor 69.5). PubMed indexed. COPE member. DOI: 10.1038/s41586-020-1234-5.",
            "Cell Reports (Clarivate JCR IF: 9.9). Indexed in PubMed, Scopus, Web of Science. Open access (CC-BY).",
            "PNAS (PubMed indexed, COPE member). Peer review: 3 months. Published 2023. DOAJ listed.",
            "eLife (DOAJ, PubMed, Scopus). Open peer review with transparent review history.",
            "Lancet Oncology. Clarivate IF: 51.1. COPE member. Standard peer review (3-6 months). PubMed PMID: 34567890.",
            "Science Advances. AAAS publication. PubMed indexed, Scopus, Web of Science. Impact Factor: 14.1.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        # RED FLAGS
        # 1. Predatory journal name patterns — binary
        has_predatory_name = any(p.search(text_lower) for p in self.predatory_re)
        features.append(1.0 if has_predatory_name else 0.0)

        # 2. Unrealistically fast review — binary
        has_fast = bool(re.search(r'(?:accept|review|publish).{0,20}(?:1|2|3)\s*(?:day|hour)|same[\s-]?day\s*(?:review|accept)', text_lower))
        features.append(1.0 if has_fast else 0.0)

        # 3. No indexing information — binary
        has_indexing = bool(re.search(r'pubmed|scopus|web\s*of\s*science|clarivate|doaj|medline|embase', text_lower))
        features.append(1.0 if not has_indexing else 0.0)

        # 4. APC-focused language — binary
        has_apc_focus = bool(re.search(r'(?:article|publication)\s*(?:processing|publication)\s*(?:charge|fee)|pay\s*to\s*publish|waiver\s*available', text_lower))
        features.append(1.0 if has_apc_focus and not has_indexing else 0.0)

        # 5. Spam/solicitation markers — binary
        has_spam = bool(re.search(r'dear\s*(?:author|researcher|professor)|submit\s*your\s*(?:paper|manuscript)|special\s*discount', text_lower))
        features.append(1.0 if has_spam else 0.0)

        # QUALITY SIGNALS
        # 6. PubMed indexed — binary
        has_pubmed = bool(re.search(r'pubmed|pmid|medline', text_lower))
        features.append(1.0 if has_pubmed else 0.0)

        # 7. Impact factor from Clarivate — binary
        has_if = bool(re.search(r'impact\s*factor|clarivate|jcr|web\s*of\s*science', text_lower))
        features.append(1.0 if has_if else 0.0)

        # 8. COPE member — binary
        has_cope = bool(re.search(r'cope\s*member|committee\s*on\s*publication\s*ethics', text_lower))
        features.append(1.0 if has_cope else 0.0)

        # 9. DOAJ listed — binary
        has_doaj = bool(re.search(r'doaj|directory\s*of\s*open\s*access', text_lower))
        features.append(1.0 if has_doaj else 0.0)

        # 10. Established publisher — binary
        has_publisher = bool(re.search(r'elsevier|springer|wiley|nature\s*(?:publishing|portfolio)|aaas|oxford|cambridge|bmc|cell\s*press', text_lower))
        features.append(1.0 if has_publisher else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> CitationPatternAntibodyResult:
        text_lower = text.lower()

        has_predatory = any(p.search(text_lower) for p in self.predatory_re)
        if has_predatory:
            return CitationPatternAntibodyResult(
                component="PredatoryJournal", is_anomaly=True, confidence=0.85,
                reason="Journal name matches predatory publisher patterns",
            )

        has_fast = bool(re.search(r'(?:accept|review).{0,20}(?:1|2|3)\s*(?:day|hour)|same[\s-]?day', text_lower))
        if has_fast:
            return CitationPatternAntibodyResult(
                component="PredatoryJournal", is_anomaly=True, confidence=0.8,
                reason="Unrealistically fast peer review timeline",
            )

        return CitationPatternAntibodyResult(
            component="PredatoryJournal", is_anomaly=False, confidence=0.6,
            reason="No predatory journal markers detected",
        )


# ---------------------------------------------------------------------------
# Combined System
# ---------------------------------------------------------------------------

class CitationPatternAntibodySystem:
    """
    Multi-antibody system for citation pattern analysis.

    Complements CitationAntibodySystem (field-level validation) with
    pattern-level checks: retraction status, self-citation, predatory journals.
    """

    def __init__(self):
        self.antibodies: Dict[str, BaseCitationPatternAntibody] = {
            "retraction": RetractionAntibody(),
            "self_citation": SelfCitationAntibody(),
            "predatory_journal": PredatoryJournalAntibody(),
        }
        self.fusion = ImmuneSignalFusion(domain="citation")

    def train_antibody(self, component: str, valid_examples: List[str]):
        if component not in self.antibodies:
            raise ValueError(f"Unknown component: {component}. Valid: {list(self.antibodies.keys())}")
        self.antibodies[component].train(valid_examples)

    def verify_citation_patterns(self, text: str) -> CitationPatternResult:
        """Verify citation patterns using all antibodies."""
        results: Dict[str, CitationPatternAntibodyResult] = {}
        anomaly_count = 0
        total_checks = 0
        bindings = []

        for component, antibody in self.antibodies.items():
            result = antibody.check(text)
            results[component] = result
            total_checks += 1
            bindings.append(result.binding_affinity)
            if result.is_anomaly:
                anomaly_count += 1

        if total_checks == 0:
            response = ImmuneResponse.REJECT
            overall_confidence = 1.0
        else:
            fusion_result = self.fusion.fuse_signals(bindings)
            overall_confidence = fusion_result.fused_binding
            response = fusion_result.response

        is_suspicious = response != ImmuneResponse.IGNORE

        return CitationPatternResult(
            is_suspicious=is_suspicious,
            overall_confidence=overall_confidence,
            response=response,
            component_results=results,
            anomaly_count=anomaly_count,
            total_checks=total_checks,
        )

    def save_all(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for name, antibody in self.antibodies.items():
            antibody.save_state(str(path / f"{name}_citpattern_antibody.pkl"))

    def load_all(self, directory: str):
        path = Path(directory)
        for name in self.antibodies.keys():
            antibody_path = path / f"{name}_citpattern_antibody.pkl"
            if antibody_path.exists():
                try:
                    with open(antibody_path, "rb") as f:
                        state = pickle.load(f)
                except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError, EOFError) as e:
                    print(f"Warning: Could not load {antibody_path}: {e}", file=sys.stderr)
                    continue
                self.antibodies[name].patterns = state.get("patterns", [])
                self.antibodies[name].is_trained = state.get("is_trained", False)
                if state.get("nk_detector"):
                    self.antibodies[name].nk_detector = state["nk_detector"]

    def get_training_status(self) -> Dict[str, bool]:
        return {name: ab.is_trained for name, ab in self.antibodies.items()}


def create_citation_pattern_antibody_system() -> CitationPatternAntibodySystem:
    """Create a new citation pattern antibody system."""
    return CitationPatternAntibodySystem()
