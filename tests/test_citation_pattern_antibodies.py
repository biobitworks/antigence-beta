"""Tests for CitationPatternAntibodySystem — 3 citation pattern analysis antibodies."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from immunos_mcp.agents.citation_pattern_antibodies import (
    CitationPatternAntibodySystem,
    RetractionAntibody,
    SelfCitationAntibody,
    PredatoryJournalAntibody,
    create_citation_pattern_antibody_system,
)
from immunos_mcp.core.immune_response import ImmuneResponse


# -- System-Level Tests -------------------------------------------------------


class TestCitationPatternAntibodySystem:
    def test_create_system(self):
        system = CitationPatternAntibodySystem()
        assert len(system.antibodies) == 3
        assert set(system.antibodies.keys()) == {
            "retraction", "self_citation", "predatory_journal",
        }

    def test_convenience_constructor(self):
        system = create_citation_pattern_antibody_system()
        assert isinstance(system, CitationPatternAntibodySystem)

    def test_training_status_initially_untrained(self):
        system = CitationPatternAntibodySystem()
        status = system.get_training_status()
        assert all(v is False for v in status.values())

    def test_clean_citations_pass(self):
        system = CitationPatternAntibodySystem()
        text = (
            "Smith 2020, Jones 2019, Zhang 2021, Kumar 2018, Lee 2022, "
            "Anderson 2020, Brown 2017, Garcia 2019, Muller 2021, Johnson 2023. "
            "Published in Nature, Science, Cell, PNAS, Lancet. PubMed indexed."
        )
        result = system.verify_citation_patterns(text)
        assert result.total_checks == 3

    def test_empty_text_flagged(self):
        system = CitationPatternAntibodySystem()
        result = system.verify_citation_patterns("")
        assert result.anomaly_count == 3

    def test_to_dict(self):
        system = CitationPatternAntibodySystem()
        result = system.verify_citation_patterns("Smith 2020. Nature. PubMed indexed.")
        d = result.to_dict()
        assert "is_suspicious" in d
        assert "response" in d
        assert d["response"] in ("ignore", "review", "reject")

    def test_response_field_exposed(self):
        system = CitationPatternAntibodySystem()
        result = system.verify_citation_patterns("Normal citation text.")
        assert isinstance(result.response, ImmuneResponse)

    def test_train_unknown_component_raises(self):
        system = CitationPatternAntibodySystem()
        with pytest.raises(ValueError, match="Unknown component"):
            system.train_antibody("impact_factor", ["test"])


# -- RetractionAntibody -------------------------------------------------------


class TestRetractionAntibody:
    def test_normal_citation_passes(self):
        ab = RetractionAntibody()
        result = ab.check("Smith J et al. (2020) Nature 580:123-130. DOI: 10.1038/s41586-020-1234-5.")
        assert result.is_anomaly is False

    def test_retracted_citation_flagged(self):
        ab = RetractionAntibody()
        result = ab.check("This paper has been retracted due to data fabrication concerns.")
        assert result.is_anomaly is True

    def test_expression_of_concern_flagged(self):
        ab = RetractionAntibody()
        result = ab.check("An expression of concern has been issued for this article.")
        assert result.is_anomaly is True

    def test_withdrawn_flagged(self):
        ab = RetractionAntibody()
        result = ab.check("This article was withdrawn by the authors.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = RetractionAntibody()
        features = ab.extract_features("Normal citation. Nature 2020. DOI: 10.1038/x.")
        assert isinstance(features, np.ndarray)
        assert len(features) == 10


# -- SelfCitationAntibody -----------------------------------------------------


class TestSelfCitationAntibody:
    def test_diverse_citations_pass(self):
        ab = SelfCitationAntibody()
        text = (
            "Smith 2020, Jones 2019, Zhang 2021, Kumar 2018, Lee 2022, "
            "Anderson 2020, Brown 2017, Garcia 2019, Muller 2021, Johnson 2023."
        )
        result = ab.check(text)
        assert result.is_anomaly is False

    def test_excessive_self_citation_flagged(self):
        ab = SelfCitationAntibody()
        text = (
            "Smith 2020, Smith 2019, Smith 2021, Smith 2018, Smith 2022, "
            "Smith 2017, Smith 2016, Jones 2020, Kumar 2019, Lee 2021."
        )
        result = ab.check(text)
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = SelfCitationAntibody()
        features = ab.extract_features("Smith 2020, Jones 2019, Zhang 2021.")
        assert len(features) == 10


# -- PredatoryJournalAntibody --------------------------------------------------


class TestPredatoryJournalAntibody:
    def test_reputable_journal_passes(self):
        ab = PredatoryJournalAntibody()
        result = ab.check("Published in Nature (Impact Factor 69.5). PubMed indexed. COPE member.")
        assert result.is_anomaly is False

    def test_predatory_pattern_flagged(self):
        ab = PredatoryJournalAntibody()
        result = ab.check("International Journal of Advanced Scientific Research. Submit your paper today!")
        assert result.is_anomaly is True

    def test_fast_review_flagged(self):
        ab = PredatoryJournalAntibody()
        result = ab.check("Paper accepted within 2 days of submission. Rapid publication guaranteed.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = PredatoryJournalAntibody()
        features = ab.extract_features("Published in Cell. PubMed. Clarivate JCR.")
        assert len(features) == 10


# -- Discrimination Tests -----------------------------------------------------


class TestCitationPatternDiscrimination:
    """Test that the system discriminates between clean and problematic citation patterns."""

    CLEAN_TEXTS = [
        "Smith 2020, Jones 2019, Zhang 2021, Kumar 2018, Lee 2022, Anderson 2020, Brown 2017, Garcia 2019. Published in Nature, Science, Cell. PubMed indexed. COPE member.",
        "References from 2010-2024. 15 different research groups. Nature, PNAS, Lancet, BMJ, JAMA. DOI: 10.1038/s41586-020-1234-5. PubMed indexed. No retractions.",
    ]

    PROBLEMATIC_TEXTS = [
        "This retracted paper from the International Journal of Advanced Innovative Research was accepted within 1 day.",
        "Smith 2020, Smith 2019, Smith 2021, Smith 2018, Smith 2022, Smith 2017. Published in Global Journal of Scientific Research. Paper retracted.",
    ]

    def test_clean_fewer_flags(self):
        system = CitationPatternAntibodySystem()
        for text in self.CLEAN_TEXTS:
            result = system.verify_citation_patterns(text)
            assert result.anomaly_count <= 1, f"Too many flags: {text[:60]}..."

    def test_problematic_more_flags(self):
        system = CitationPatternAntibodySystem()
        for text in self.PROBLEMATIC_TEXTS:
            result = system.verify_citation_patterns(text)
            assert result.anomaly_count >= 2, f"Too few flags: {text[:60]}..."
