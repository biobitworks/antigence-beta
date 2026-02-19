#!/usr/bin/env python3
"""
Figure Integrity Antibody System - Multi-Antibody Architecture for Figure/Caption Verification
================================================================================================
Each figure component (caption completeness, representative claims, quantification,
panel consistency, data presentation, image source) has its own specialized antibody.

Text-based analysis of figure captions/legends — not pixel-level image analysis.
Detects missing statistical context, cherry-picked figures, poor data visualization practices.
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
class FigureAntibodyResult:
    """Result from a single figure integrity antibody check."""
    component: str
    is_anomaly: bool
    confidence: float
    matched_pattern: Optional[str] = None
    reason: str = ""
    binding_affinity: float = 0.0


@dataclass
class FigureIntegrityResult:
    """Combined result from all figure integrity antibodies."""
    is_suspicious: bool
    overall_confidence: float
    response: ImmuneResponse = ImmuneResponse.IGNORE
    component_results: Dict[str, FigureAntibodyResult] = field(default_factory=dict)
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


class BaseFigureAntibody:
    """Base class for figure integrity antibodies."""

    def __init__(self, component_name: str, num_detectors: int = 50):
        self.component_name = component_name
        self.patterns: List[str] = []
        self.config = NegSelConfig(
            num_detectors=num_detectors,
            r_self=0.85,
            description=f"{component_name} Figure Antibody",
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

    def check(self, text: str) -> FigureAntibodyResult:
        if not text or not text.strip():
            return FigureAntibodyResult(
                component=self.component_name,
                is_anomaly=True, confidence=1.0, binding_affinity=1.0,
                reason="Empty or missing caption",
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

        return FigureAntibodyResult(
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

    def _rule_based_check(self, text: str) -> FigureAntibodyResult:
        return FigureAntibodyResult(
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
    def load_state(cls, path: str) -> "BaseFigureAntibody":
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
# Antibody 1: Caption Completeness Antibody
# ---------------------------------------------------------------------------

class CaptionCompletenessAntibody(BaseFigureAntibody):
    """
    Detects incomplete figure captions: no n, no error bars, no scale bar, no units.

    Red flags: No n, no error bars, no scale bar, no units
    Quality signals: Statistical test named, p-values, source data
    """

    def __init__(self):
        super().__init__("CaptionCompleteness", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "Figure 1. Western blot analysis of protein expression (n=3 biological replicates). Error bars represent SEM. *p<0.05 by Student's t-test. Scale bar: 50 um.",
            "Figure 2. Dose-response curve for compound X (n=6 per group). Data shown as mean +/- SD. Statistical analysis by one-way ANOVA with Tukey post-hoc. Source data in Supplementary Table 1.",
            "Figure 3. Confocal microscopy images of GFP-tagged cells (n=4 independent experiments). Scale bar: 10 um. Quantification in panel B (mean +/- SEM, n=50 cells per condition).",
            "Figure 4. Survival curve (Kaplan-Meier) for treatment vs control (n=30 per group). p=0.003 by log-rank test. Hazard ratio 0.45 (95% CI: 0.28-0.72).",
            "Figure 5. (A) Representative flow cytometry plots (n=5). (B) Quantification of CD4+ T cells as percentage of live cells. Error bars: SD. **p<0.01, unpaired t-test.",
            "Figure 6. Metabolomics heatmap (n=8 per group). Color scale: log2 fold change. Clustering: Ward's method. Significant metabolites (FDR < 0.05) marked with asterisks.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        # RED FLAGS
        # 1. No sample size (n=) — binary
        has_n = bool(re.search(r'n\s*=\s*\d|n\s*=\s*\w|sample\s*size|replicates?|per\s*(?:group|condition)', text_lower))
        features.append(1.0 if not has_n else 0.0)

        # 2. No error bars/uncertainty — binary
        has_error = bool(re.search(r'error\s*bar|sem|s\.?d\.?|s\.?e\.?m\.?|standard\s*(?:deviation|error)|confidence\s*interval|\+/?-|±', text_lower))
        features.append(1.0 if not has_error else 0.0)

        # 3. No statistical test — binary
        has_stat = bool(re.search(r't[\s-]?test|anova|mann[\s-]?whitney|chi[\s-]?sq|wilcoxon|log[\s-]?rank|fisher|kruskal|bonferroni|tukey', text_lower))
        features.append(1.0 if not has_stat else 0.0)

        # 4. No units or scale bar for microscopy — binary
        is_microscopy = bool(re.search(r'microscop|confocal|fluorescen|histolog|immunostain|staining|image', text_lower))
        has_scale = bool(re.search(r'scale\s*bar|[uμ]m|mm|nm|magnification', text_lower))
        features.append(1.0 if is_microscopy and not has_scale else 0.0)

        # 5. No p-value for comparison figures — binary
        is_comparison = bool(re.search(r'vs\.?|versus|compar|between|treatment|control', text_lower))
        has_p = bool(re.search(r'p\s*[=<>]|p[\s-]?value|n\.?s\.?|not\s*significant|\*p', text_lower))
        features.append(1.0 if is_comparison and not has_p else 0.0)

        # QUALITY SIGNALS
        # 6. Statistical test named — binary
        features.append(1.0 if has_stat else 0.0)

        # 7. P-values reported — binary
        features.append(1.0 if has_p else 0.0)

        # 8. Source data referenced — binary
        has_source = bool(re.search(r'source\s*data|supplementary|raw\s*data|data\s*(?:available|deposited)', text_lower))
        features.append(1.0 if has_source else 0.0)

        # 9. Sample size reported — binary
        features.append(1.0 if has_n else 0.0)

        # 10. Error bars described — binary
        features.append(1.0 if has_error else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> FigureAntibodyResult:
        text_lower = text.lower()
        has_n = bool(re.search(r'n\s*=\s*\d|replicates?|per\s*(?:group|condition)', text_lower))
        has_error = bool(re.search(r'error\s*bar|sem|s\.?d\.?|s\.?e\.?m\.?|\+/?-|±', text_lower))
        has_stat = bool(re.search(r't[\s-]?test|anova|mann[\s-]?whitney|wilcoxon|log[\s-]?rank', text_lower))

        missing = []
        if not has_n:
            missing.append("sample size")
        if not has_error:
            missing.append("error bars/uncertainty")
        if not has_stat:
            missing.append("statistical test")

        if len(missing) >= 2:
            return FigureAntibodyResult(
                component="CaptionCompleteness", is_anomaly=True, confidence=0.8,
                reason=f"Caption missing: {', '.join(missing)}",
            )

        return FigureAntibodyResult(
            component="CaptionCompleteness", is_anomaly=False, confidence=0.7,
            reason="Caption appears complete",
        )


# ---------------------------------------------------------------------------
# Antibody 2: Representative Claim Antibody
# ---------------------------------------------------------------------------

class RepresentativeClaimAntibody(BaseFigureAntibody):
    """
    Detects cherry-picked figures: "representative" without n, "typical", no quantification.

    Red flags: "Representative" without n, "typical" without context
    Quality signals: "n=X experiments", all replicates shown
    """

    def __init__(self):
        super().__init__("RepresentativeClaim", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "Representative images from n=5 independent experiments. All replicates shown in Supplementary Figure 2.",
            "Representative of 3 biological replicates with similar results. Quantification in panel C.",
            "Images representative of n=10 mice per group. Full dataset in Supplementary Data.",
            "One representative experiment of four is shown. Quantitative analysis of all experiments in Figure 3B.",
            "Data from one representative donor of n=6 donors tested. Individual donor data in Supplementary Figure 1.",
            "Representative flow cytometry plots (n=8). Summary statistics for all samples in panel D.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        has_representative = bool(re.search(r'representative|typical|example\s*(?:image|blot|gel)', text_lower))
        has_n_context = bool(re.search(r'(?:representative|typical).{0,50}n\s*=\s*\d|n\s*=\s*\d.{0,50}(?:representative|typical)', text_lower))

        # RED FLAGS
        # 1. "Representative" without n — binary
        features.append(1.0 if has_representative and not has_n_context else 0.0)

        # 2. "Typical" without quantification — binary
        has_typical = bool(re.search(r'\btypical\b', text_lower))
        has_quant = bool(re.search(r'quantif|all\s*(?:replicates|samples|data)|summary|mean|average', text_lower))
        features.append(1.0 if has_typical and not has_quant else 0.0)

        # 3. Cherry-pick language — binary
        has_cherry = bool(re.search(r'best\s*(?:result|example|image)|selected\s*(?:image|example)', text_lower))
        features.append(1.0 if has_cherry else 0.0)

        # 4. Single replicate shown without justification — binary
        has_single = bool(re.search(r'one\s*(?:experiment|replicate|sample)|single\s*(?:experiment|example)', text_lower))
        has_justification = bool(re.search(r'similar\s*result|all\s*replicate|full\s*data|supplementary', text_lower))
        features.append(1.0 if has_single and not has_justification else 0.0)

        # 5. No quantification for qualitative image — binary
        is_qualitative = bool(re.search(r'image|blot|gel|microscop|stain|photograph', text_lower))
        features.append(1.0 if is_qualitative and not has_quant else 0.0)

        # QUALITY SIGNALS
        # 6. n= context for representative — binary
        features.append(1.0 if has_n_context else 0.0)

        # 7. All replicates shown — binary
        has_all = bool(re.search(r'all\s*(?:replicates|samples|data\s*shown)|individual\s*(?:data|values)', text_lower))
        features.append(1.0 if has_all else 0.0)

        # 8. Quantification referenced — binary
        features.append(1.0 if has_quant else 0.0)

        # 9. Supplementary data referenced — binary
        has_supp = bool(re.search(r'supplementary|suppl|source\s*data|full\s*dataset', text_lower))
        features.append(1.0 if has_supp else 0.0)

        # 10. Multiple experiments stated — binary
        has_multiple = bool(re.search(r'(?:three|four|five|\d)\s*(?:independent|biological|separate)\s*(?:experiment|replicate)', text_lower))
        features.append(1.0 if has_multiple else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> FigureAntibodyResult:
        text_lower = text.lower()
        has_representative = bool(re.search(r'representative|typical', text_lower))
        has_n = bool(re.search(r'n\s*=\s*\d|replicates?\s*\d|\d\s*(?:independent|biological)', text_lower))

        if has_representative and not has_n:
            return FigureAntibodyResult(
                component="RepresentativeClaim", is_anomaly=True, confidence=0.8,
                reason="'Representative' image without sample size context",
            )

        return FigureAntibodyResult(
            component="RepresentativeClaim", is_anomaly=False, confidence=0.7,
            reason="Representative claims appear substantiated",
        )


# ---------------------------------------------------------------------------
# Antibody 3: Quantification Antibody
# ---------------------------------------------------------------------------

class QuantificationAntibody(BaseFigureAntibody):
    """
    Detects missing quantification: "significant" without p-value, fold-change without raw data.

    Red flags: "Significant" without p, fold-change without raw data
    Quality signals: Individual data points, CIs, effect size
    """

    def __init__(self):
        super().__init__("Quantification", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "Fold change = 2.3 (95% CI: 1.8-2.9, p=0.001). Individual data points shown. Raw data in Source Data file.",
            "Significant increase (p=0.003, Cohen's d=0.8). Error bars: SEM. Each dot represents one biological replicate.",
            "Mean difference: 15.2 units (95% CI: 10.1-20.3). Individual participant data overlaid on bar chart.",
            "Quantification of band intensity normalized to GAPDH (n=4). Values expressed as fold change over control. *p<0.05, **p<0.01.",
            "Box plots show median and interquartile range. Individual data points (circles). p-values by Kruskal-Wallis with Dunn's post-hoc.",
            "Effect size (Hedges' g = 0.65, 95% CI: 0.22-1.08). Raw values in Supplementary Table 3.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        # RED FLAGS
        # 1. "Significant" without p-value — binary
        has_significant = bool(re.search(r'\bsignificant\b', text_lower))
        has_p = bool(re.search(r'p\s*[=<>]|p[\s-]?value|\*p|n\.?s\.?', text_lower))
        features.append(1.0 if has_significant and not has_p else 0.0)

        # 2. Fold-change without raw data reference — binary
        has_fold = bool(re.search(r'fold[\s-]?change|fold\s*(?:increase|decrease)', text_lower))
        has_raw = bool(re.search(r'raw\s*data|source\s*data|individual\s*(?:data|values|points)', text_lower))
        features.append(1.0 if has_fold and not has_raw else 0.0)

        # 3. No effect size for comparison — binary
        is_comparison = bool(re.search(r'(?:increase|decrease|higher|lower|greater|reduced|compared)', text_lower))
        has_effect = bool(re.search(r'effect\s*size|cohen|hedges|eta|odds\s*ratio|risk\s*ratio|hazard\s*ratio', text_lower))
        features.append(1.0 if is_comparison and not has_effect and not has_p else 0.0)

        # 4. Percentage without denominator — binary
        has_percent = bool(re.search(r'\d+\s*%', text_lower))
        has_denominator = bool(re.search(r'of\s*\d+|out\s*of|n\s*=|total', text_lower))
        features.append(1.0 if has_percent and not has_denominator else 0.0)

        # 5. Vague magnitude claims — binary
        has_vague = bool(re.search(r'(?:dramatic|striking|robust|marked|pronounced|modest)\s*(?:increase|decrease|change|difference|effect)', text_lower))
        features.append(1.0 if has_vague and not has_p else 0.0)

        # QUALITY SIGNALS
        # 6. Individual data points shown — binary
        has_individual = bool(re.search(r'individual\s*(?:data|values|points)|each\s*(?:dot|point|circle)|overlaid', text_lower))
        features.append(1.0 if has_individual else 0.0)

        # 7. Confidence intervals — binary
        has_ci = bool(re.search(r'confidence\s*interval|95\s*%\s*ci|\bci\b|ci:', text_lower))
        features.append(1.0 if has_ci else 0.0)

        # 8. Effect size reported — binary
        features.append(1.0 if has_effect else 0.0)

        # 9. P-value reported — binary
        features.append(1.0 if has_p else 0.0)

        # 10. Raw/source data referenced — binary
        features.append(1.0 if has_raw else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> FigureAntibodyResult:
        text_lower = text.lower()
        has_significant = bool(re.search(r'\bsignificant\b', text_lower))
        has_p = bool(re.search(r'p\s*[=<>]|p[\s-]?value|\*p', text_lower))

        if has_significant and not has_p:
            return FigureAntibodyResult(
                component="Quantification", is_anomaly=True, confidence=0.85,
                reason="'Significant' claimed without p-value",
            )

        return FigureAntibodyResult(
            component="Quantification", is_anomaly=False, confidence=0.7,
            reason="Quantification appears adequate",
        )


# ---------------------------------------------------------------------------
# Antibody 4: Panel Consistency Antibody
# ---------------------------------------------------------------------------

class PanelConsistencyAntibody(BaseFigureAntibody):
    """
    Detects panel inconsistency: inconsistent axes, missing panel references, duplicate labels.

    Red flags: Inconsistent axes, missing panel refs, duplicate labels
    Quality signals: All panels referenced, consistent scheme
    """

    def __init__(self):
        super().__init__("PanelConsistency", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "(A) Western blot. (B) Quantification of blot in A. (C) mRNA expression. All panels: n=3, error bars = SEM.",
            "Panels A-D show individual biomarkers. Panel E shows composite score. Same y-axis scale used across A-D. Color coding consistent throughout.",
            "(A) Control cells. (B) Treatment 1. (C) Treatment 2. (D) Quantification of A-C. All images at same magnification (40x).",
            "Figure panels referenced in text as (A)-(F). (A,B) Histology. (C,D) Immunofluorescence. (E,F) Quantification of C,D.",
            "Left panels (A,C,E): male subjects. Right panels (B,D,F): female subjects. Axes matched for direct comparison.",
            "All subpanels (A-H) use the same color scheme: blue = control, red = treatment, green = combination.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        # Extract panel references
        panels = re.findall(r'(?:panel\s*)?([A-H])\b', text, re.IGNORECASE)
        panel_set = set(p.upper() for p in panels)

        # RED FLAGS
        # 1. Panel labels referenced but gaps in sequence — binary
        if len(panel_set) >= 2:
            expected = set(chr(i) for i in range(ord(min(panel_set)), ord(max(panel_set)) + 1))
            has_gap = expected - panel_set
            features.append(1.0 if has_gap else 0.0)
        else:
            features.append(0.0)

        # 2. Duplicate panel labels — binary
        if panels:
            from collections import Counter
            counts = Counter(p.upper() for p in panels)
            has_dup = any(c > 3 for c in counts.values())  # Allow some repetition in text
            features.append(1.0 if has_dup else 0.0)
        else:
            features.append(0.0)

        # 3. Multi-panel figure without cross-references — binary
        has_multi = len(panel_set) >= 3
        has_cross_ref = bool(re.search(r'(?:panel|figure|of|from|in)\s+[A-H](?:\s+(?:and|,)\s+[A-H])?|[A-H]\s*(?:and|,)\s*[A-H]|[A-H]\s*-\s*[A-H]', text, re.IGNORECASE))
        features.append(1.0 if has_multi and not has_cross_ref else 0.0)

        # 4. No consistent axis/scale mention for comparison — binary
        is_comparison = bool(re.search(r'compar|same\s*(?:scale|axis|magnification)|matched', text_lower))
        features.append(1.0 if has_multi and not is_comparison else 0.0)

        # 5. No color/symbol legend for multi-group — binary
        has_multi_group = bool(re.search(r'(?:blue|red|green|black|white|filled|open|circle|square|triangle)', text_lower))
        has_legend = bool(re.search(r'(?:color|colour)\s*(?:code|scheme|legend)|legend|symbol', text_lower))
        features.append(1.0 if has_multi_group and not has_legend and has_multi else 0.0)

        # QUALITY SIGNALS
        # 6. All panels referenced — binary
        features.append(1.0 if has_multi and has_cross_ref else 0.0)

        # 7. Consistent scheme stated — binary
        has_consistent = bool(re.search(r'consistent|same\s*(?:scale|axis|color|magnification)|throughout|across\s*(?:all|panels)', text_lower))
        features.append(1.0 if has_consistent else 0.0)

        # 8. Panel descriptions present — binary
        has_descriptions = len(panel_set) >= 2
        features.append(1.0 if has_descriptions else 0.0)

        # 9. Axes/units described — binary
        has_axes = bool(re.search(r'(?:x|y)[\s-]?axis|(?:axis|axes)\s*(?:label|show|represent)', text_lower))
        features.append(1.0 if has_axes else 0.0)

        # 10. Magnification/scale consistent — binary
        has_mag = bool(re.search(r'magnification|scale\s*bar|(?:same|identical)\s*(?:magnification|scale)', text_lower))
        features.append(1.0 if has_mag else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> FigureAntibodyResult:
        panels = set(re.findall(r'(?:panel\s*)?([A-H])\b', text, re.IGNORECASE))

        if len(panels) >= 3:
            has_cross_ref = bool(re.search(r'(?:panel|figure|of|from|in)\s+[A-H](?:\s+(?:and|,)\s+[A-H])?|[A-H]\s*(?:and|,)\s*[A-H]|[A-H]\s*-\s*[A-H]', text, re.IGNORECASE))
            if not has_cross_ref:
                return FigureAntibodyResult(
                    component="PanelConsistency", is_anomaly=True, confidence=0.75,
                    reason="Multi-panel figure without cross-referencing between panels",
                )

        return FigureAntibodyResult(
            component="PanelConsistency", is_anomaly=False, confidence=0.6,
            reason="Panel consistency appears acceptable",
        )


# ---------------------------------------------------------------------------
# Antibody 5: Data Presentation Antibody
# ---------------------------------------------------------------------------

class DataPresentationAntibody(BaseFigureAntibody):
    """
    Detects poor data presentation: bar charts without data points, 3D charts, truncated axes.

    Red flags: Bar chart without data points, 3D chart, truncated axis
    Quality signals: Violin/box plot, log scale, color-blind friendly
    """

    def __init__(self):
        super().__init__("DataPresentation", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "Box plot with individual data points overlaid. Whiskers extend to 1.5x IQR. Color-blind friendly palette (viridis).",
            "Violin plot showing full distribution. Median and quartiles indicated. Each dot = one biological replicate.",
            "Scatter plot with linear regression line (solid) and 95% CI (shaded). Log scale on y-axis due to skewed distribution.",
            "SuperPlot showing individual data points grouped by replicate (color) with mean +/- SEM of replicate means.",
            "Forest plot of effect sizes with 95% CIs. Diamond = pooled estimate. Log scale for odds ratios.",
            "Dot plot with mean and SEM. All individual values shown. Y-axis starts at 0.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        # RED FLAGS
        # 1. Bar chart without individual data points — binary
        has_bar = bool(re.search(r'bar\s*(?:chart|graph|plot)|bar\s*(?:represent|show)', text_lower))
        has_data_points = bool(re.search(r'individual\s*(?:data|values|points)|each\s*(?:dot|point)|overlaid|superimposed|jitter', text_lower))
        features.append(1.0 if has_bar and not has_data_points else 0.0)

        # 2. 3D chart — binary
        has_3d = bool(re.search(r'3[\s-]?d\s*(?:chart|graph|plot|bar|pie)|three[\s-]?dimensional\s*(?:chart|graph|plot)', text_lower))
        features.append(1.0 if has_3d else 0.0)

        # 3. Truncated/broken axis without note — binary
        has_truncated = bool(re.search(r'truncat|broken\s*axis|axis\s*break|(?:y|x)[\s-]?axis\s*(?:start|begin)', text_lower))
        has_note = bool(re.search(r'note|indicated|break\s*(?:mark|symbol|indicated)', text_lower))
        features.append(1.0 if has_truncated and not has_note else 0.0)

        # 4. Pie chart for continuous data — binary
        has_pie = bool(re.search(r'pie\s*(?:chart|graph)', text_lower))
        features.append(1.0 if has_pie else 0.0)

        # 5. No axis labels — binary
        has_axis_info = bool(re.search(r'(?:x|y)[\s-]?axis|axis\s*label|(?:unit|label)', text_lower))
        has_figure_label = bool(re.search(r'^figure|^fig', text_lower))
        features.append(1.0 if has_figure_label and not has_axis_info and len(text) < 100 else 0.0)

        # QUALITY SIGNALS
        # 6. Violin/box/dot plot — binary
        has_good_plot = bool(re.search(r'violin|box\s*plot|dot\s*plot|beeswarm|superplot|strip\s*plot|jitter', text_lower))
        features.append(1.0 if has_good_plot else 0.0)

        # 7. Log scale when appropriate — binary
        has_log = bool(re.search(r'log\s*(?:scale|transform|axis)|log\d|logarithm', text_lower))
        features.append(1.0 if has_log else 0.0)

        # 8. Color-blind friendly — binary
        has_accessible = bool(re.search(r'color[\s-]?blind|viridis|cividis|accessible|colorbrewer|palette', text_lower))
        features.append(1.0 if has_accessible else 0.0)

        # 9. Individual data points shown — binary
        features.append(1.0 if has_data_points else 0.0)

        # 10. Y-axis starts at zero — binary
        has_zero = bool(re.search(r'(?:y[\s-]?axis|axis)\s*(?:starts?\s*at|from)\s*(?:zero|0)', text_lower))
        features.append(1.0 if has_zero else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> FigureAntibodyResult:
        text_lower = text.lower()
        has_bar = bool(re.search(r'bar\s*(?:chart|graph|plot)', text_lower))
        has_data_points = bool(re.search(r'individual\s*(?:data|values|points)|each\s*(?:dot|point)|overlaid', text_lower))

        if has_bar and not has_data_points:
            return FigureAntibodyResult(
                component="DataPresentation", is_anomaly=True, confidence=0.75,
                reason="Bar chart without individual data points",
            )

        has_3d = bool(re.search(r'3[\s-]?d\s*(?:chart|graph|plot|bar)', text_lower))
        if has_3d:
            return FigureAntibodyResult(
                component="DataPresentation", is_anomaly=True, confidence=0.8,
                reason="3D chart obscures data interpretation",
            )

        return FigureAntibodyResult(
            component="DataPresentation", is_anomaly=False, confidence=0.6,
            reason="Data presentation appears appropriate",
        )


# ---------------------------------------------------------------------------
# Antibody 6: Image Source Antibody
# ---------------------------------------------------------------------------

class ImageSourceAntibody(BaseFigureAntibody):
    """
    Detects image provenance issues: stock image markers, watermarks, no processing info.

    Red flags: Stock image markers, watermark, no processing info
    Quality signals: Original data stated, processing pipeline described
    """

    def __init__(self):
        super().__init__("ImageSource", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "Original confocal images acquired on Zeiss LSM 880. Processing: brightness/contrast adjusted uniformly across all panels using ImageJ v1.53.",
            "Images captured with Nikon Eclipse microscope (40x objective). Raw .tiff files available upon request. No post-processing beyond cropping.",
            "Gel images acquired with Bio-Rad ChemiDoc XRS+. Full uncropped blots in Supplementary Figure 10. Linear adjustments only.",
            "Flow cytometry data acquired on BD FACSAria III. Analysis: FlowJo v10.8. Gating strategy in Supplementary Figure 1.",
            "Histological images from H&E-stained sections. Digital scanning: Hamamatsu NanoZoomer. No digital modifications.",
            "Fluorescence microscopy images. Equipment: Leica DMi8. Deconvolution: Huygens. All panels processed identically.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        # RED FLAGS
        # 1. Stock image markers — binary
        has_stock = bool(re.search(r'stock\s*(?:image|photo)|shutterstock|getty|istock|adobe\s*stock|dreamstime|123rf', text_lower))
        features.append(1.0 if has_stock else 0.0)

        # 2. Watermark mentioned — binary
        has_watermark = bool(re.search(r'watermark|copyright\s*mark|logo\s*overlay', text_lower))
        features.append(1.0 if has_watermark else 0.0)

        # 3. No processing/acquisition info — binary
        has_processing = bool(re.search(r'(?:acquired|captured|imaged|scanned)\s*(?:on|with|using)|processing|adjustment|software|microscope|camera|scanner', text_lower))
        is_image_fig = bool(re.search(r'image|microscop|blot|gel|histolog|fluorescen|stain|photograph|scan', text_lower))
        features.append(1.0 if is_image_fig and not has_processing else 0.0)

        # 4. "Downloaded from" without attribution — binary
        has_downloaded = bool(re.search(r'downloaded\s*from|taken\s*from\s*(?:the\s*)?internet', text_lower))
        features.append(1.0 if has_downloaded else 0.0)

        # 5. AI-generated image markers — binary
        has_ai = bool(re.search(r'(?:generated|created)\s*(?:by|with|using)\s*(?:ai|dall[\s-]?e|midjourney|stable\s*diffusion)', text_lower))
        features.append(1.0 if has_ai else 0.0)

        # QUALITY SIGNALS
        # 6. Original data stated — binary
        has_original = bool(re.search(r'original\s*(?:data|image)|raw\s*(?:data|file|image)|unprocessed|uncropped', text_lower))
        features.append(1.0 if has_original else 0.0)

        # 7. Equipment specified — binary
        has_equipment = bool(re.search(r'zeiss|nikon|leica|olympus|bio[\s-]?rad|hamamatsu|bd\s*facs|perkin[\s-]?elmer|thermo', text_lower))
        features.append(1.0 if has_equipment else 0.0)

        # 8. Software named — binary
        has_software = bool(re.search(r'imagej|fiji|photoshop|flowjo|prism|matlab|cellprofiler|ilastik|metamorph', text_lower))
        features.append(1.0 if has_software else 0.0)

        # 9. Processing described — binary
        features.append(1.0 if has_processing else 0.0)

        # 10. Full blots/gels available — binary
        has_full = bool(re.search(r'full\s*(?:blot|gel|uncropped|unprocessed)|supplementary.{0,30}(?:blot|gel)', text_lower))
        features.append(1.0 if has_full else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> FigureAntibodyResult:
        text_lower = text.lower()
        has_stock = bool(re.search(r'stock\s*(?:image|photo)|shutterstock|getty|istock', text_lower))

        if has_stock:
            return FigureAntibodyResult(
                component="ImageSource", is_anomaly=True, confidence=0.9,
                reason="Stock image markers detected",
            )

        return FigureAntibodyResult(
            component="ImageSource", is_anomaly=False, confidence=0.6,
            reason="Image source appears original",
        )


# ---------------------------------------------------------------------------
# Combined System
# ---------------------------------------------------------------------------

class FigureIntegrityAntibodySystem:
    """
    Multi-antibody system for comprehensive figure/caption verification.

    Text-based analysis of figure captions and legends for completeness,
    representative claims, quantification, panel consistency, data presentation,
    and image provenance.
    """

    def __init__(self):
        self.antibodies: Dict[str, BaseFigureAntibody] = {
            "caption_completeness": CaptionCompletenessAntibody(),
            "representative_claim": RepresentativeClaimAntibody(),
            "quantification": QuantificationAntibody(),
            "panel_consistency": PanelConsistencyAntibody(),
            "data_presentation": DataPresentationAntibody(),
            "image_source": ImageSourceAntibody(),
        }
        self.fusion = ImmuneSignalFusion(domain="analysis")

    def train_antibody(self, component: str, valid_examples: List[str]):
        if component not in self.antibodies:
            raise ValueError(f"Unknown component: {component}. Valid: {list(self.antibodies.keys())}")
        self.antibodies[component].train(valid_examples)

    def verify_figures(self, caption: str) -> FigureIntegrityResult:
        """Verify a figure caption using all antibodies."""
        results: Dict[str, FigureAntibodyResult] = {}
        anomaly_count = 0
        total_checks = 0
        bindings = []

        for component, antibody in self.antibodies.items():
            result = antibody.check(caption)
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

        return FigureIntegrityResult(
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
            antibody.save_state(str(path / f"{name}_figure_antibody.pkl"))

    def load_all(self, directory: str):
        path = Path(directory)
        for name in self.antibodies.keys():
            antibody_path = path / f"{name}_figure_antibody.pkl"
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


def create_figure_integrity_antibody_system() -> FigureIntegrityAntibodySystem:
    """Create a new figure integrity antibody system."""
    return FigureIntegrityAntibodySystem()
