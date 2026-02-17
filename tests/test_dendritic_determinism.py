"""
Falsification tests for Dendritic Agent determinism.

Core claims:
- Same input always produces the same feature vector
- Feature vector is always 20 floats
- Feature values are bounded and meaningful
"""

import pytest

from immunos_mcp.core.antigen import Antigen
from immunos_mcp.agents.dendritic_agent import DendriticAgent


@pytest.fixture
def agent():
    return DendriticAgent()


@pytest.fixture
def sample_text_antigen():
    return Antigen.from_text(
        "According to Smith et al. (2024), the treatment cures 100% of patients. "
        "This breakthrough is revolutionary and unprecedented."
    )


@pytest.fixture
def sample_code_antigen():
    return Antigen.from_code("eval(input())")


@pytest.fixture
def safe_text_antigen():
    return Antigen.from_text(
        "The study suggests that exercise may improve cardiovascular health. "
        "Further research is needed to confirm these findings (DOI: 10.1234/test)."
    )


class TestDeterminism:
    """Same input -> same output, every time."""

    def test_same_input_same_vector_100_runs(self, agent, sample_text_antigen):
        """FALSIFICATION: If any run differs, the feature extraction is non-deterministic."""
        reference = agent.get_feature_vector(sample_text_antigen)
        for i in range(100):
            result = agent.get_feature_vector(sample_text_antigen)
            assert result == reference, (
                f"Run {i}: feature vector differs from reference. "
                f"Dendritic features are NOT deterministic."
            )

    def test_same_code_same_vector(self, agent, sample_code_antigen):
        reference = agent.get_feature_vector(sample_code_antigen)
        for _ in range(50):
            assert agent.get_feature_vector(sample_code_antigen) == reference


class TestFeatureVectorShape:
    """Feature vector must always be exactly 20 floats."""

    def test_vector_length_20(self, agent, sample_text_antigen):
        vector = agent.get_feature_vector(sample_text_antigen)
        assert len(vector) == 20

    def test_vector_length_for_code(self, agent, sample_code_antigen):
        assert len(agent.get_feature_vector(sample_code_antigen)) == 20

    def test_vector_length_for_empty_text(self, agent):
        empty = Antigen.from_text("")
        assert len(agent.get_feature_vector(empty)) == 20

    def test_all_elements_are_float(self, agent, sample_text_antigen):
        vector = agent.get_feature_vector(sample_text_antigen)
        for i, v in enumerate(vector):
            assert isinstance(v, float), f"Element {i} is {type(v)}, not float"


class TestFeatureSemantics:
    """Features should capture meaningful properties."""

    def test_danger_signals_detected(self, agent, sample_text_antigen):
        """Text with 'cures 100%' should trigger danger signals."""
        features = agent.extract_features(sample_text_antigen)
        assert features["pamp_score"] > 0
        assert features["danger_signal_count"] > 0

    def test_citation_detected(self, agent, safe_text_antigen):
        """Text with DOI should be flagged as having citations."""
        features = agent.extract_features(safe_text_antigen)
        assert features["has_citation"] is True

    def test_hedging_detected(self, agent, safe_text_antigen):
        """Text with 'suggests', 'may' should flag hedging."""
        features = agent.extract_features(safe_text_antigen)
        assert features["has_hedging"] is True

    def test_safe_text_higher_credibility(self, agent, sample_text_antigen, safe_text_antigen):
        """Safe scientific text should have higher credibility than danger text."""
        danger_features = agent.extract_features(sample_text_antigen)
        safe_features = agent.extract_features(safe_text_antigen)
        assert safe_features["source_credibility"] > danger_features["source_credibility"]

    def test_exaggeration_detected(self, agent, sample_text_antigen):
        """Text with 'revolutionary', 'unprecedented' should have exaggeration."""
        features = agent.extract_features(sample_text_antigen)
        assert features["exaggeration_score"] > 0


class TestSignalClassification:
    """Test the classify_signals method."""

    def test_danger_signal_classification(self, agent):
        """Text with multiple danger patterns should classify as DANGER."""
        extreme = Antigen.from_text(
            "This miracle cure guarantees 100% success with no side effects."
        )
        features = agent.extract_features(extreme)
        classification = agent.classify_signals(features)
        assert classification["signal_type"] == "DANGER"

    def test_safe_signal_classification(self, agent, safe_text_antigen):
        features = agent.extract_features(safe_text_antigen)
        classification = agent.classify_signals(features)
        assert classification["signal_type"] == "SAFE"
