"""Tests for MethodologyAntibodySystem — 6 methodology verification antibodies."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from immunos_mcp.agents.methodology_antibodies import (
    MethodologyAntibodySystem,
    SampleSizeAntibody,
    BlindingAntibody,
    ControlGroupAntibody,
    PreregistrationAntibody,
    RandomizationAntibody,
    InclusionCriteriaAntibody,
    create_methodology_antibody_system,
)
from immunos_mcp.core.immune_response import ImmuneResponse


# -- System-Level Tests -------------------------------------------------------


class TestMethodologyAntibodySystem:
    def test_create_system(self):
        system = MethodologyAntibodySystem()
        assert len(system.antibodies) == 6
        assert set(system.antibodies.keys()) == {
            "sample_size", "blinding", "control_group",
            "preregistration", "randomization", "inclusion_criteria",
        }

    def test_convenience_constructor(self):
        system = create_methodology_antibody_system()
        assert isinstance(system, MethodologyAntibodySystem)

    def test_training_status_initially_untrained(self):
        system = MethodologyAntibodySystem()
        status = system.get_training_status()
        assert all(v is False for v in status.values())

    def test_good_methods_section_passes(self):
        system = MethodologyAntibodySystem()
        text = (
            "This double-blind, placebo-controlled, randomized trial enrolled 120 participants "
            "(60 per group). Sample size was determined by power analysis (alpha=0.05, power=0.80). "
            "Computer-generated block randomization with allocation concealment via sealed envelopes. "
            "Registered at ClinicalTrials.gov (NCT01234567). Inclusion criteria: age 18-65, "
            "diagnosed with type 2 diabetes (HbA1c > 6.5%). Exclusion: pregnancy, eGFR < 30."
        )
        result = system.verify_methodology(text)
        assert result.is_suspicious is False
        assert result.total_checks == 6

    def test_poor_methods_flagged(self):
        system = MethodologyAntibodySystem()
        text = "We gave the drug to some patients and they got better."
        result = system.verify_methodology(text)
        assert result.anomaly_count >= 2

    def test_empty_text_flagged(self):
        system = MethodologyAntibodySystem()
        result = system.verify_methodology("")
        assert result.anomaly_count == 6

    def test_to_dict(self):
        system = MethodologyAntibodySystem()
        result = system.verify_methodology("A double-blind randomized trial with n=50.")
        d = result.to_dict()
        assert "is_suspicious" in d
        assert "overall_confidence" in d
        assert "response" in d
        assert d["response"] in ("ignore", "review", "reject")
        assert "components" in d

    def test_response_field_exposed(self):
        system = MethodologyAntibodySystem()
        result = system.verify_methodology("A double-blind randomized trial with n=50.")
        assert isinstance(result.response, ImmuneResponse)

    def test_train_single_antibody(self):
        system = MethodologyAntibodySystem()
        examples = [
            "N=100 participants enrolled",
            "N=50 per group",
            "Sample size: 200 patients",
        ]
        system.train_antibody("sample_size", examples)
        status = system.get_training_status()
        assert status["sample_size"] is True
        assert status["blinding"] is False

    def test_train_unknown_component_raises(self):
        system = MethodologyAntibodySystem()
        with pytest.raises(ValueError, match="Unknown component"):
            system.train_antibody("abstract", ["test"])


# -- SampleSizeAntibody -------------------------------------------------------


class TestSampleSizeAntibody:
    def test_adequate_sample_passes(self):
        ab = SampleSizeAntibody()
        result = ab.check("We enrolled 120 participants (60 per group). Power analysis showed n=50 was sufficient.")
        assert result.is_anomaly is False

    def test_missing_sample_size_flagged(self):
        ab = SampleSizeAntibody()
        result = ab.check("We treated the clinical patients and measured outcomes.")
        assert result.is_anomaly is True

    def test_tiny_sample_flagged(self):
        ab = SampleSizeAntibody()
        result = ab.check("n=5 patients were enrolled in this clinical trial.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = SampleSizeAntibody()
        features = ab.extract_features("N=100 participants with power analysis")
        assert isinstance(features, np.ndarray)
        assert len(features) == 10

    def test_empty_is_anomaly(self):
        ab = SampleSizeAntibody()
        result = ab.check("")
        assert result.is_anomaly is True


# -- BlindingAntibody ----------------------------------------------------------


class TestBlindingAntibody:
    def test_double_blind_passes(self):
        ab = BlindingAntibody()
        result = ab.check("Double-blind, placebo-controlled trial with allocation concealment.")
        assert result.is_anomaly is False

    def test_rct_without_blinding_flagged(self):
        ab = BlindingAntibody()
        result = ab.check("This randomized controlled trial assigned patients to groups.")
        assert result.is_anomaly is True

    def test_open_label_with_justification_passes(self):
        ab = BlindingAntibody()
        result = ab.check("This was an open-label trial due to the nature of the surgical intervention.")
        assert result.is_anomaly is False

    def test_feature_extraction(self):
        ab = BlindingAntibody()
        features = ab.extract_features("Double-blind design with independent assessors")
        assert len(features) == 10


# -- ControlGroupAntibody -----------------------------------------------------


class TestControlGroupAntibody:
    def test_placebo_control_passes(self):
        ab = ControlGroupAntibody()
        result = ab.check("Participants were randomized to treatment or placebo control group.")
        assert result.is_anomaly is False

    def test_no_control_flagged(self):
        ab = ControlGroupAntibody()
        result = ab.check("All patients received the experimental treatment and were monitored.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = ControlGroupAntibody()
        features = ab.extract_features("Matched case-control design")
        assert len(features) == 10


# -- PreregistrationAntibody ---------------------------------------------------


class TestPreregistrationAntibody:
    def test_registered_trial_passes(self):
        ab = PreregistrationAntibody()
        result = ab.check("Registered at ClinicalTrials.gov (NCT01234567) before enrollment.")
        assert result.is_anomaly is False

    def test_clinical_trial_without_registration_flagged(self):
        ab = PreregistrationAntibody()
        result = ab.check("This randomized clinical trial enrolled 200 patients.")
        assert result.is_anomaly is True

    def test_posthoc_as_primary_flagged(self):
        ab = PreregistrationAntibody()
        result = ab.check("Post-hoc analysis was used as the primary outcome measure.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = PreregistrationAntibody()
        features = ab.extract_features("Pre-registered on OSF (osf.io/abc123)")
        assert len(features) == 10


# -- RandomizationAntibody ----------------------------------------------------


class TestRandomizationAntibody:
    def test_computer_randomization_passes(self):
        ab = RandomizationAntibody()
        result = ab.check("Computer-generated block randomization with stratification by age and sex.")
        assert result.is_anomaly is False

    def test_randomization_without_method_flagged(self):
        ab = RandomizationAntibody()
        result = ab.check("Patients were randomized to two groups.")
        assert result.is_anomaly is True

    def test_alternation_flagged(self):
        ab = RandomizationAntibody()
        result = ab.check("Patients were assigned by alternation to treatment groups.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = RandomizationAntibody()
        features = ab.extract_features("Permuted block randomization")
        assert len(features) == 10


# -- InclusionCriteriaAntibody ------------------------------------------------


class TestInclusionCriteriaAntibody:
    def test_specific_criteria_passes(self):
        ab = InclusionCriteriaAntibody()
        result = ab.check("Inclusion criteria: age 18-65, HbA1c > 6.5%, BMI 25-40. Exclusion: pregnancy, eGFR < 30.")
        assert result.is_anomaly is False

    def test_no_criteria_flagged(self):
        ab = InclusionCriteriaAntibody()
        result = ab.check("We analyzed data from the hospital database.")
        assert result.is_anomaly is True

    def test_criteria_changed_flagged(self):
        ab = InclusionCriteriaAntibody()
        result = ab.check("The eligibility criteria were changed during the study to increase enrollment.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = InclusionCriteriaAntibody()
        features = ab.extract_features("Inclusion criteria: DSM-5 diagnosis, HAMD >= 17")
        assert len(features) == 10


# -- Discrimination Tests -----------------------------------------------------


class TestMethodologyDiscrimination:
    """Test that the system discriminates between good and poor methods."""

    GOOD_METHODS = [
        "This double-blind, placebo-controlled, randomized trial enrolled 120 participants (60 per group). Sample size was determined by power analysis. Computer-generated block randomization. Registered at ClinicalTrials.gov (NCT01234567). Inclusion criteria: age 18-65, diagnosed with type 2 diabetes.",
        "A single-blind randomized controlled trial with n=200, placebo comparator group. Stratified randomization by age and sex. Allocation concealment via central pharmacy. Pre-registered on OSF (osf.io/abc123). Inclusion criteria: adults with confirmed COVID-19 by PCR. Exclusion: severe comorbidities.",
    ]

    POOR_METHODS = [
        "We gave the drug to some patients and they got better.",
        "The clinical trial administered treatment. Results were positive. The study confirms efficacy.",
    ]

    def test_good_methods_not_suspicious(self):
        system = MethodologyAntibodySystem()
        for text in self.GOOD_METHODS:
            result = system.verify_methodology(text)
            assert result.anomaly_count <= 2, f"Too many flags on good methods: {text[:60]}..."

    def test_poor_methods_flagged(self):
        system = MethodologyAntibodySystem()
        for text in self.POOR_METHODS:
            result = system.verify_methodology(text)
            assert result.anomaly_count >= 2, f"Too few flags on poor methods: {text[:60]}..."

    def test_anomaly_separation(self):
        system = MethodologyAntibodySystem()
        good_counts = [system.verify_methodology(t).anomaly_count for t in self.GOOD_METHODS]
        poor_counts = [system.verify_methodology(t).anomaly_count for t in self.POOR_METHODS]
        assert max(good_counts) <= min(poor_counts)
