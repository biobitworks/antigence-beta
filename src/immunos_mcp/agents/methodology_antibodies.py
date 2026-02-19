#!/usr/bin/env python3
"""
Methodology Antibody System - Multi-Antibody Architecture for Methods Section Verification
============================================================================================
Each methodology component (sample size, blinding, control group, preregistration,
randomization, inclusion criteria) has its own specialized antibody trained on real patterns.

Detects methodological red flags in study methods sections: missing controls,
no blinding, unreported sample sizes, absent preregistration, weak randomization.
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
class MethodologyAntibodyResult:
    """Result from a single methodology antibody check."""
    component: str
    is_anomaly: bool
    confidence: float
    matched_pattern: Optional[str] = None
    reason: str = ""
    binding_affinity: float = 0.0


@dataclass
class MethodologyVerificationResult:
    """Combined result from all methodology antibodies."""
    is_suspicious: bool
    overall_confidence: float
    response: ImmuneResponse = ImmuneResponse.IGNORE
    component_results: Dict[str, MethodologyAntibodyResult] = field(default_factory=dict)
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


class BaseMethodologyAntibody:
    """Base class for methodology verification antibodies."""

    def __init__(self, component_name: str, num_detectors: int = 50):
        self.component_name = component_name
        self.patterns: List[str] = []
        self.config = NegSelConfig(
            num_detectors=num_detectors,
            r_self=0.85,
            description=f"{component_name} Methodology Antibody",
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

    def check(self, text: str) -> MethodologyAntibodyResult:
        if not text or not text.strip():
            return MethodologyAntibodyResult(
                component=self.component_name,
                is_anomaly=True,
                confidence=1.0,
                binding_affinity=1.0,
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

        return MethodologyAntibodyResult(
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
                [self.extract_features(v) for v in self_examples],
                dtype=np.float32,
            )
            self.nk_detector.fit(self_features)
            self.is_trained = True
            return self.nk_detector.get_anomaly_score(features)
        return 0.5

    def _generate_self_examples(self) -> List[str]:
        return []

    def _rule_based_check(self, text: str) -> MethodologyAntibodyResult:
        return MethodologyAntibodyResult(
            component=self.component_name,
            is_anomaly=False,
            confidence=0.5,
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
    def load_state(cls, path: str) -> "BaseMethodologyAntibody":
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
# Antibody 1: Sample Size Antibody
# ---------------------------------------------------------------------------

class SampleSizeAntibody(BaseMethodologyAntibody):
    """
    Detects sample size red flags: unreported n, tiny samples for human studies,
    round-number bias suggesting fabrication.

    Red flags: n<10 for human study, no n reported, round-number bias
    Quality signals: Power analysis, n>30, attrition reported
    """

    def __init__(self):
        super().__init__("SampleSize", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "A total of 120 participants were enrolled (60 per group). Sample size was determined by power analysis (alpha=0.05, power=0.80, effect size d=0.5). Attrition: 8 participants dropped out (3 treatment, 5 control).",
            "We recruited 45 healthy volunteers (age 25-65). Power calculation indicated n=40 required per group. All 45 completed the study.",
            "The study enrolled 200 patients meeting inclusion criteria. A priori sample size calculation based on previous effect sizes indicated n=180 needed. Final analysis included 192 participants after exclusions.",
            "Thirty-two participants (16 male, 16 female) completed all sessions. This sample size was based on a power analysis using G*Power with alpha=0.05 and power=0.80.",
            "We analyzed data from 500 subjects in the cohort study. Attrition rate was 12% over the 5-year follow-up period.",
            "N=85 patients were randomized. Sample size was calculated to detect a 15% difference with 80% power. CONSORT flow diagram shows 78 completed.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        # Extract sample sizes
        n_values = [int(m) for m in re.findall(r'\bn\s*=\s*(\d+)', text_lower)]
        n_values += [int(m) for m in re.findall(r'(\d+)\s*(?:participants|patients|subjects|volunteers)', text_lower)]

        # RED FLAGS
        # 1. Very small sample (n<10) mentioned — binary
        has_tiny = any(n < 10 for n in n_values) if n_values else False
        features.append(1.0 if has_tiny else 0.0)

        # 2. No sample size reported at all — binary
        has_any_n = bool(n_values) or bool(re.search(r'\bn\s*=|sample\s*size|participants|subjects', text_lower))
        features.append(1.0 if not has_any_n else 0.0)

        # 3. Round-number bias (all n values are multiples of 10) — binary
        if len(n_values) >= 2:
            all_round = all(n % 10 == 0 for n in n_values)
            features.append(1.0 if all_round else 0.0)
        else:
            features.append(0.0)

        # 4. Human study keywords without sample size — binary
        is_human = bool(re.search(r'patient|participant|volunteer|subject|cohort|clinical|trial', text_lower))
        features.append(1.0 if is_human and not has_any_n else 0.0)

        # 5. Single-digit sample for clinical claim — binary
        has_clinical = bool(re.search(r'clinical|efficacy|treatment|therapeutic', text_lower))
        features.append(1.0 if has_clinical and has_tiny else 0.0)

        # QUALITY SIGNALS
        # 6. Has power analysis — binary
        has_power = bool(re.search(r'power\s*(?:analysis|calculation|calc)|g\*power|sample\s*size\s*(?:calculation|determination)', text_lower))
        features.append(1.0 if has_power else 0.0)

        # 7. Adequate sample (n>30) — binary
        has_adequate = any(n > 30 for n in n_values) if n_values else False
        features.append(1.0 if has_adequate else 0.0)

        # 8. Reports attrition/dropout — binary
        has_attrition = bool(re.search(r'attrition|dropout|drop.?out|withdrew|lost\s*to\s*follow', text_lower))
        features.append(1.0 if has_attrition else 0.0)

        # 9. CONSORT flow mentioned — binary
        has_consort = bool(re.search(r'consort|flow\s*diagram|strobe', text_lower))
        features.append(1.0 if has_consort else 0.0)

        # 10. Reports per-group n — binary
        has_per_group = bool(re.search(r'per\s*group|each\s*group|per\s*arm|\d+\s*per', text_lower))
        features.append(1.0 if has_per_group else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> MethodologyAntibodyResult:
        text_lower = text.lower()
        n_values = [int(m) for m in re.findall(r'\bn\s*=\s*(\d+)', text_lower)]
        n_values += [int(m) for m in re.findall(r'(\d+)\s*(?:participants|patients|subjects)', text_lower)]

        is_human = bool(re.search(r'patient|participant|volunteer|subject|clinical|trial', text_lower))
        has_any_n = bool(n_values) or bool(re.search(r'\bn\s*=|sample\s*size', text_lower))

        if is_human and not has_any_n:
            return MethodologyAntibodyResult(
                component="SampleSize", is_anomaly=True, confidence=0.85,
                reason="Human study without reported sample size",
            )

        if n_values and any(n < 10 for n in n_values) and is_human:
            return MethodologyAntibodyResult(
                component="SampleSize", is_anomaly=True, confidence=0.8,
                reason=f"Very small sample size (n={min(n_values)}) for human study",
            )

        return MethodologyAntibodyResult(
            component="SampleSize", is_anomaly=False, confidence=0.7,
            reason="Sample size reporting appears adequate",
        )


# ---------------------------------------------------------------------------
# Antibody 2: Blinding Antibody
# ---------------------------------------------------------------------------

class BlindingAntibody(BaseMethodologyAntibody):
    """
    Detects blinding red flags: RCT without blinding, open-label without justification.

    Red flags: RCT without blinding, open-label no justification
    Quality signals: Double-blind, allocation concealment
    """

    def __init__(self):
        super().__init__("Blinding", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "This double-blind, placebo-controlled trial used matching placebo capsules. Neither participants nor investigators knew group assignments until data lock.",
            "Allocation concealment was maintained using sequentially numbered, opaque, sealed envelopes. Outcome assessors were blinded to group assignment.",
            "Single-blind design: participants were unaware of their group assignment. Blinding of outcome assessment was performed by independent evaluators.",
            "This was an open-label trial due to the nature of the surgical intervention. However, outcome assessors and data analysts were blinded to group allocation.",
            "Double-blind, double-dummy design. Active drug and placebo were identical in appearance, taste, and packaging.",
            "Triple-blind study: participants, investigators, and statisticians were blinded until the final analysis was unblocked per protocol.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        is_rct = bool(re.search(r'randomized|randomised|rct|controlled\s*trial', text_lower))
        has_blinding = bool(re.search(r'blind|blinding|masked|masking', text_lower))
        is_open_label = bool(re.search(r'open[\s-]?label|unblinded|not\s*blinded', text_lower))

        # RED FLAGS
        # 1. RCT without any blinding mention — binary
        features.append(1.0 if is_rct and not has_blinding and not is_open_label else 0.0)

        # 2. Open-label without justification — binary
        has_justification = bool(re.search(r'due\s*to|because|nature\s*of|not\s*feasible|impractical', text_lower))
        features.append(1.0 if is_open_label and not has_justification else 0.0)

        # 3. Subjective outcome without blinding — binary
        has_subjective = bool(re.search(r'patient[\s-]?reported|subjective|self[\s-]?report|pain\s*scale|quality\s*of\s*life', text_lower))
        features.append(1.0 if has_subjective and not has_blinding else 0.0)

        # 4. No allocation concealment in RCT — binary
        has_concealment = bool(re.search(r'allocation\s*concealment|sealed\s*envelope|central\s*randomization|pharmacy[\s-]?controlled', text_lower))
        features.append(1.0 if is_rct and not has_concealment else 0.0)

        # 5. Unblinded assessors — binary
        unblinded_assessors = bool(re.search(r'assessor.{0,20}not\s*blind|unblinded\s*(?:assessor|evaluator|rater)', text_lower))
        features.append(1.0 if unblinded_assessors else 0.0)

        # QUALITY SIGNALS
        # 6. Double-blind — binary
        is_double_blind = bool(re.search(r'double[\s-]?blind|double[\s-]?mask', text_lower))
        features.append(1.0 if is_double_blind else 0.0)

        # 7. Allocation concealment — binary
        features.append(1.0 if has_concealment else 0.0)

        # 8. Blinding described in detail — binary
        has_detail = bool(re.search(r'identical\s*(?:in\s*)?(?:appearance|packaging|capsule)|matching\s*placebo|indistinguishable', text_lower))
        features.append(1.0 if has_detail else 0.0)

        # 9. Blinding assessment/verification — binary
        has_verification = bool(re.search(r'blinding\s*(?:assessment|index|success|verified|tested)', text_lower))
        features.append(1.0 if has_verification else 0.0)

        # 10. Independent assessors — binary
        has_independent = bool(re.search(r'independent\s*(?:assessor|evaluator|reviewer)|assessor.{0,20}blind', text_lower))
        features.append(1.0 if has_independent else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> MethodologyAntibodyResult:
        text_lower = text.lower()
        is_rct = bool(re.search(r'randomized|randomised|rct|controlled\s*trial', text_lower))
        has_blinding = bool(re.search(r'blind|blinding|masked|masking', text_lower))
        is_open_label = bool(re.search(r'open[\s-]?label|unblinded', text_lower))

        if is_rct and not has_blinding and not is_open_label:
            return MethodologyAntibodyResult(
                component="Blinding", is_anomaly=True, confidence=0.85,
                reason="Randomized controlled trial without blinding description",
            )

        if is_open_label and not bool(re.search(r'due\s*to|because|nature\s*of', text_lower)):
            return MethodologyAntibodyResult(
                component="Blinding", is_anomaly=True, confidence=0.75,
                reason="Open-label design without justification",
            )

        return MethodologyAntibodyResult(
            component="Blinding", is_anomaly=False, confidence=0.7,
            reason="Blinding description appears adequate",
        )


# ---------------------------------------------------------------------------
# Antibody 3: Control Group Antibody
# ---------------------------------------------------------------------------

class ControlGroupAntibody(BaseMethodologyAntibody):
    """
    Detects control group red flags: no control, historical only, self-control only.

    Red flags: No control, historical controls only, self-control without washout
    Quality signals: Matched/placebo/active control, randomized allocation
    """

    def __init__(self):
        super().__init__("ControlGroup", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "Participants were randomly assigned to treatment (n=60) or placebo control (n=60). The placebo was identical in appearance.",
            "We used an active control group receiving standard-of-care treatment. Both groups were matched for age, sex, and disease severity.",
            "A parallel-group design with 1:1 randomization to intervention or wait-list control was used.",
            "Three groups: active treatment, active comparator, and placebo control. Randomization was stratified by site.",
            "Matched case-control design with 2 controls per case, matched on age, sex, and BMI.",
            "Crossover design with 4-week washout period between treatment and placebo phases.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        has_control = bool(re.search(r'control\s*group|placebo|comparator|comparison\s*group|wait[\s-]?list|sham', text_lower))
        is_comparative = bool(re.search(r'compar|versus|vs\.?|between\s*groups|group\s*(?:a|b|1|2)', text_lower))

        # RED FLAGS
        # 1. No control group mentioned — binary
        features.append(1.0 if not has_control and not is_comparative else 0.0)

        # 2. Historical controls only — binary
        has_historical = bool(re.search(r'historical\s*control|compared\s*(?:to|with)\s*(?:previous|historical|published)', text_lower))
        features.append(1.0 if has_historical and not has_control else 0.0)

        # 3. Self-control without washout — binary
        has_self_control = bool(re.search(r'self[\s-]?control|before[\s-]?after|pre[\s-]?post', text_lower))
        has_washout = bool(re.search(r'washout|wash[\s-]?out|recovery\s*period', text_lower))
        features.append(1.0 if has_self_control and not has_washout else 0.0)

        # 4. Single-arm study for treatment claim — binary
        has_single_arm = bool(re.search(r'single[\s-]?arm|one[\s-]?arm|uncontrolled', text_lower))
        has_treatment_claim = bool(re.search(r'efficacy|effective|superior|treatment\s*effect', text_lower))
        features.append(1.0 if has_single_arm and has_treatment_claim else 0.0)

        # 5. No randomization to groups — binary
        has_randomization = bool(re.search(r'random|randomiz|randomis', text_lower))
        features.append(1.0 if is_comparative and not has_randomization and not has_control else 0.0)

        # QUALITY SIGNALS
        # 6. Placebo or sham control — binary
        has_placebo = bool(re.search(r'placebo|sham', text_lower))
        features.append(1.0 if has_placebo else 0.0)

        # 7. Matched controls — binary
        has_matched = bool(re.search(r'matched|matching|stratified|balanced', text_lower))
        features.append(1.0 if has_matched else 0.0)

        # 8. Active comparator — binary
        has_active = bool(re.search(r'active\s*(?:control|comparator)|standard[\s-]?of[\s-]?care|current\s*treatment', text_lower))
        features.append(1.0 if has_active else 0.0)

        # 9. Multiple comparison groups — binary
        has_multiple = bool(re.search(r'three\s*groups|four\s*groups|multiple\s*(?:arms|groups)|3[\s-]?arm|4[\s-]?arm', text_lower))
        features.append(1.0 if has_multiple else 0.0)

        # 10. Crossover design — binary
        has_crossover = bool(re.search(r'crossover|cross[\s-]?over|latin\s*square', text_lower))
        features.append(1.0 if has_crossover else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> MethodologyAntibodyResult:
        text_lower = text.lower()
        has_control = bool(re.search(r'control\s*group|placebo|comparator|comparison\s*group|sham', text_lower))
        is_comparative = bool(re.search(r'compar|versus|vs\.?|between\s*groups', text_lower))

        if not has_control and not is_comparative:
            return MethodologyAntibodyResult(
                component="ControlGroup", is_anomaly=True, confidence=0.8,
                reason="No control or comparison group described",
            )

        return MethodologyAntibodyResult(
            component="ControlGroup", is_anomaly=False, confidence=0.7,
            reason="Control group description present",
        )


# ---------------------------------------------------------------------------
# Antibody 4: Preregistration Antibody
# ---------------------------------------------------------------------------

class PreregistrationAntibody(BaseMethodologyAntibody):
    """
    Detects preregistration red flags: no registration for clinical trial,
    post-hoc analyses presented as primary.

    Red flags: No prereg for clinical trial, post-hoc as primary
    Quality signals: NCT/ISRCTN registry, OSF link, deviation disclosed
    """

    def __init__(self):
        super().__init__("Preregistration", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "This trial was registered at ClinicalTrials.gov (NCT01234567) before enrollment began. Protocol deviations are reported in the supplement.",
            "Pre-registered on the Open Science Framework (osf.io/abc123). All hypotheses and analysis plans were specified a priori.",
            "Registered with ISRCTN (ISRCTN12345678). Primary and secondary outcomes were pre-specified. Post-hoc exploratory analyses are clearly labeled.",
            "Study protocol was published before data collection (DOI: 10.1186/protocol-2020-001). Pre-analysis plan available at osf.io/xyz.",
            "Prospectively registered on ClinicalTrials.gov (NCT98765432). Primary endpoint: overall survival at 2 years. Secondary endpoints pre-specified.",
            "PROSPERO registration CRD42020123456. Systematic review protocol followed PRISMA-P guidelines.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        is_clinical = bool(re.search(r'clinical\s*trial|rct|randomized|randomised|interventional', text_lower))
        has_registry = bool(re.search(r'nct\d|isrctn\d|prospero|clinicaltrials\.gov|osf\.io|aspredicted', text_lower))

        # RED FLAGS
        # 1. Clinical trial without registration — binary
        features.append(1.0 if is_clinical and not has_registry else 0.0)

        # 2. Post-hoc presented as primary — binary
        has_posthoc_primary = bool(re.search(r'post[\s-]?hoc.{0,30}(?:primary|main|principal)', text_lower))
        features.append(1.0 if has_posthoc_primary else 0.0)

        # 3. Outcome switching signals — binary
        has_outcome_switch = bool(re.search(r'(?:changed|modified|revised)\s*(?:primary|endpoint|outcome)', text_lower))
        features.append(1.0 if has_outcome_switch and not bool(re.search(r'disclosed|reported|documented', text_lower)) else 0.0)

        # 4. Exploratory analysis without label — binary
        has_exploratory = bool(re.search(r'explor|subgroup|secondary', text_lower))
        has_label = bool(re.search(r'exploratory|post[\s-]?hoc|hypothesis[\s-]?generating|not\s*pre[\s-]?specified', text_lower))
        features.append(1.0 if has_exploratory and not has_label and is_clinical else 0.0)

        # 5. Multiple primary endpoints without justification — binary
        primary_count = len(re.findall(r'primary\s*(?:endpoint|outcome|measure)', text_lower))
        features.append(1.0 if primary_count >= 3 else 0.0)

        # QUALITY SIGNALS
        # 6. Has registry ID — binary
        features.append(1.0 if has_registry else 0.0)

        # 7. OSF or pre-analysis plan — binary
        has_plan = bool(re.search(r'osf\.io|pre[\s-]?analysis\s*plan|aspredicted|pre[\s-]?registered|a\s*priori', text_lower))
        features.append(1.0 if has_plan else 0.0)

        # 8. Protocol deviations disclosed — binary
        has_deviations = bool(re.search(r'(?:protocol|pre[\s-]?registration)\s*deviation|deviation.{0,20}(?:reported|disclosed)', text_lower))
        features.append(1.0 if has_deviations else 0.0)

        # 9. Prospective registration — binary
        has_prospective = bool(re.search(r'prospect|before\s*enrollment|before\s*recruitment|prior\s*to\s*(?:enrollment|data)', text_lower))
        features.append(1.0 if has_prospective else 0.0)

        # 10. Published protocol — binary
        has_protocol = bool(re.search(r'published\s*protocol|protocol\s*paper|protocol\s*(?:doi|10\.)', text_lower))
        features.append(1.0 if has_protocol else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> MethodologyAntibodyResult:
        text_lower = text.lower()
        is_clinical = bool(re.search(r'clinical\s*trial|rct|randomized|randomised', text_lower))
        has_registry = bool(re.search(r'nct\d|isrctn\d|prospero|clinicaltrials\.gov|osf\.io', text_lower))

        if is_clinical and not has_registry:
            return MethodologyAntibodyResult(
                component="Preregistration", is_anomaly=True, confidence=0.85,
                reason="Clinical trial without registration ID",
            )

        has_posthoc_primary = bool(re.search(r'post[\s-]?hoc.{0,30}(?:primary|main)', text_lower))
        if has_posthoc_primary:
            return MethodologyAntibodyResult(
                component="Preregistration", is_anomaly=True, confidence=0.8,
                reason="Post-hoc analysis presented as primary outcome",
            )

        return MethodologyAntibodyResult(
            component="Preregistration", is_anomaly=False, confidence=0.7,
            reason="Preregistration status acceptable",
        )


# ---------------------------------------------------------------------------
# Antibody 5: Randomization Antibody
# ---------------------------------------------------------------------------

class RandomizationAntibody(BaseMethodologyAntibody):
    """
    Detects randomization red flags: no method described, alternation/sequential.

    Red flags: No randomization method, alternation, sequential assignment
    Quality signals: Computer-generated, block/stratified randomization
    """

    def __init__(self):
        super().__init__("Randomization", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "Computer-generated random number sequence was used for allocation. Block randomization with variable block sizes (4, 6, 8) was employed.",
            "Stratified randomization by age and sex using a web-based system. Allocation ratio 1:1.",
            "Permuted block randomization with allocation concealment via central telephone system.",
            "Simple randomization using sealed, opaque envelopes prepared by an independent statistician.",
            "Adaptive randomization (minimization) balancing treatment arms for prognostic factors.",
            "Cluster randomization at the clinic level, with stratification by region. Random sequence generated using SAS PROC PLAN.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        has_random = bool(re.search(r'random|randomiz|randomis', text_lower))
        has_method = bool(re.search(r'computer[\s-]?generated|block\s*random|stratif|permut|minimiz|adaptive\s*random', text_lower))

        # RED FLAGS
        # 1. Claims randomization without describing method — binary
        features.append(1.0 if has_random and not has_method else 0.0)

        # 2. Alternation or sequential assignment — binary
        has_alternation = bool(re.search(r'alternat|sequential|every\s*other|odd[\s-]?even|by\s*(?:day|week|order)', text_lower))
        features.append(1.0 if has_alternation else 0.0)

        # 3. Quasi-randomization — binary
        has_quasi = bool(re.search(r'quasi[\s-]?random|pseudo[\s-]?random|convenience\s*sampl|non[\s-]?random', text_lower))
        features.append(1.0 if has_quasi else 0.0)

        # 4. No concealment with randomization — binary
        has_concealment = bool(re.search(r'conceal|sealed|opaque|central|telephone|web[\s-]?based|pharmacy', text_lower))
        features.append(1.0 if has_random and not has_concealment else 0.0)

        # 5. Unequal groups without explanation — binary
        has_unequal = bool(re.search(r'unequal|(?:2|3|4):1\s*ratio', text_lower))
        has_explanation = bool(re.search(r'because|due\s*to|in\s*order\s*to|to\s*increase\s*power', text_lower))
        features.append(1.0 if has_unequal and not has_explanation else 0.0)

        # QUALITY SIGNALS
        # 6. Computer-generated sequence — binary
        has_computer = bool(re.search(r'computer[\s-]?generated|software|sas|stata|r\s*(?:function|package)|web[\s-]?based', text_lower))
        features.append(1.0 if has_computer else 0.0)

        # 7. Block randomization — binary
        has_block = bool(re.search(r'block\s*random|permut|variable\s*block', text_lower))
        features.append(1.0 if has_block else 0.0)

        # 8. Stratified — binary
        has_stratified = bool(re.search(r'stratif|minimiz|adaptive|balanced', text_lower))
        features.append(1.0 if has_stratified else 0.0)

        # 9. Allocation concealment described — binary
        features.append(1.0 if has_concealment else 0.0)

        # 10. Independent randomization agent — binary
        has_independent = bool(re.search(r'independent\s*(?:statistician|pharmacist|center)|central\s*(?:randomiz|allocation)', text_lower))
        features.append(1.0 if has_independent else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> MethodologyAntibodyResult:
        text_lower = text.lower()
        has_random = bool(re.search(r'random|randomiz|randomis', text_lower))
        has_method = bool(re.search(r'computer[\s-]?generated|block|stratif|permut|minimiz', text_lower))

        if has_random and not has_method:
            return MethodologyAntibodyResult(
                component="Randomization", is_anomaly=True, confidence=0.8,
                reason="Randomization claimed but method not described",
            )

        has_alternation = bool(re.search(r'alternat|sequential|every\s*other', text_lower))
        if has_alternation:
            return MethodologyAntibodyResult(
                component="Randomization", is_anomaly=True, confidence=0.85,
                reason="Alternation/sequential assignment is not true randomization",
            )

        return MethodologyAntibodyResult(
            component="Randomization", is_anomaly=False, confidence=0.7,
            reason="Randomization method appears adequate",
        )


# ---------------------------------------------------------------------------
# Antibody 6: Inclusion Criteria Antibody
# ---------------------------------------------------------------------------

class InclusionCriteriaAntibody(BaseMethodologyAntibody):
    """
    Detects inclusion criteria red flags: no criteria specified, changed during study.

    Red flags: No criteria specified, criteria changed during study
    Quality signals: Specific diagnostic criteria, CONSORT flow
    """

    def __init__(self):
        super().__init__("InclusionCriteria", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "Inclusion criteria: age 18-65, diagnosed with type 2 diabetes (HbA1c > 6.5%), BMI 25-40. Exclusion: pregnancy, renal failure (eGFR < 30), active malignancy.",
            "Eligible patients met DSM-5 criteria for major depressive disorder, HAMD-17 score >= 17, and had not responded to at least one antidepressant trial.",
            "Adults (>18 years) with histologically confirmed stage III/IV non-small cell lung cancer. ECOG performance status 0-1. Adequate organ function per protocol.",
            "Inclusion: community-dwelling adults aged 65+, MMSE >= 24, ambulatory without assistive device. Exclusion: neurological conditions, unstable cardiac disease.",
            "We included studies of adults with chronic low back pain (>3 months), randomized design, and validated pain outcome measure. Excluded: surgical trials, cancer pain.",
            "Participants met CONSORT criteria. Eligibility: confirmed COVID-19 by PCR, symptom onset within 5 days, no prior vaccination. CONSORT flow diagram in Figure 1.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        has_criteria = bool(re.search(r'(?:inclusion|exclusion|eligib|eligible)\s*(?:criteria|criterion)', text_lower))
        has_specific = bool(re.search(r'(?:age|bmi|score|stage|diagnosed|confirmed|status)\s*[><=]|(?:dsm|icd|who|ecog|nyha)', text_lower))

        # RED FLAGS
        # 1. No inclusion/exclusion criteria mentioned — binary
        has_any_criteria = has_criteria or bool(re.search(r'included\s*(?:if|when|patients)|excluded\s*(?:if|patients)', text_lower))
        features.append(1.0 if not has_any_criteria else 0.0)

        # 2. Criteria changed during study — binary
        has_changed = bool(re.search(r'(?:criteria|eligibility)\s*(?:were|was)\s*(?:changed|modified|revised|amended)', text_lower))
        features.append(1.0 if has_changed else 0.0)

        # 3. Vague criteria only — binary
        has_vague = bool(re.search(r'(?:healthy|suitable|appropriate|willing)\s*(?:volunteers?|participants?|subjects?)', text_lower))
        features.append(1.0 if has_vague and not has_specific else 0.0)

        # 4. No exclusion criteria — binary
        has_exclusion = bool(re.search(r'exclusion|excluded|ineligible|not\s*eligible', text_lower))
        features.append(1.0 if has_criteria and not has_exclusion else 0.0)

        # 5. Post-hoc exclusion — binary
        has_posthoc_exclusion = bool(re.search(r'post[\s-]?hoc\s*(?:exclusion|removed|eliminated)|excluded\s*(?:after|during)\s*analysis', text_lower))
        features.append(1.0 if has_posthoc_exclusion else 0.0)

        # QUALITY SIGNALS
        # 6. Specific diagnostic criteria — binary
        features.append(1.0 if has_specific else 0.0)

        # 7. CONSORT flow diagram — binary
        has_consort = bool(re.search(r'consort|flow\s*diagram|participant\s*flow|strobe', text_lower))
        features.append(1.0 if has_consort else 0.0)

        # 8. Both inclusion AND exclusion specified — binary
        features.append(1.0 if has_criteria and has_exclusion else 0.0)

        # 9. Age range specified — binary
        has_age = bool(re.search(r'age\s*(?:\d|>|<|between|range)|(?:adults?|children|elderly|aged\s*\d)', text_lower))
        features.append(1.0 if has_age else 0.0)

        # 10. Validated assessment tool referenced — binary
        has_validated = bool(re.search(r'(?:hamd|phq|gad|mmse|moca|ecog|karnofsky|vas|sf[\s-]?36|eq[\s-]?5d|who)', text_lower))
        features.append(1.0 if has_validated else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> MethodologyAntibodyResult:
        text_lower = text.lower()
        has_criteria = bool(re.search(r'(?:inclusion|exclusion|eligib)\s*(?:criteria|criterion)', text_lower))
        has_any = has_criteria or bool(re.search(r'included\s*(?:if|when|patients)|excluded\s*(?:if|patients)', text_lower))

        if not has_any:
            return MethodologyAntibodyResult(
                component="InclusionCriteria", is_anomaly=True, confidence=0.8,
                reason="No inclusion or exclusion criteria specified",
            )

        has_changed = bool(re.search(r'(?:criteria|eligibility)\s*(?:were|was)\s*(?:changed|modified|revised)', text_lower))
        if has_changed:
            return MethodologyAntibodyResult(
                component="InclusionCriteria", is_anomaly=True, confidence=0.85,
                reason="Inclusion/exclusion criteria were changed during the study",
            )

        return MethodologyAntibodyResult(
            component="InclusionCriteria", is_anomaly=False, confidence=0.7,
            reason="Inclusion criteria appear specified",
        )


# ---------------------------------------------------------------------------
# Combined System
# ---------------------------------------------------------------------------

class MethodologyAntibodySystem:
    """
    Multi-antibody system for comprehensive methodology verification.

    Checks methods sections for sample size reporting, blinding, controls,
    preregistration, randomization, and inclusion criteria.
    """

    def __init__(self):
        self.antibodies: Dict[str, BaseMethodologyAntibody] = {
            "sample_size": SampleSizeAntibody(),
            "blinding": BlindingAntibody(),
            "control_group": ControlGroupAntibody(),
            "preregistration": PreregistrationAntibody(),
            "randomization": RandomizationAntibody(),
            "inclusion_criteria": InclusionCriteriaAntibody(),
        }
        self.fusion = ImmuneSignalFusion(domain="analysis")

    def train_antibody(self, component: str, valid_examples: List[str]):
        if component not in self.antibodies:
            raise ValueError(f"Unknown component: {component}. Valid: {list(self.antibodies.keys())}")
        self.antibodies[component].train(valid_examples)

    def verify_methodology(self, text: str) -> MethodologyVerificationResult:
        """Verify a methods section using all antibodies."""
        results: Dict[str, MethodologyAntibodyResult] = {}
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

        return MethodologyVerificationResult(
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
            antibody.save_state(str(path / f"{name}_methodology_antibody.pkl"))

    def load_all(self, directory: str):
        path = Path(directory)
        for name in self.antibodies.keys():
            antibody_path = path / f"{name}_methodology_antibody.pkl"
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


def create_methodology_antibody_system() -> MethodologyAntibodySystem:
    """Create a new methodology antibody system."""
    return MethodologyAntibodySystem()
