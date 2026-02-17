"""
Falsification tests for NegSl-AIS Negative Selection Algorithm.

Core claim: Detectors NEVER fire on self-data (R_q > R_self constraint).
"""

import numpy as np
import pytest

from immunos_mcp.algorithms.negsel import (
    NegativeSelectionClassifier,
    NegSelConfig,
    calculate_mcc,
    calculate_kappa,
)


@pytest.fixture
def clustered_self_data():
    """Self-data clustered tightly in a small region of the unit hypercube."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.1, 0.3, size=(30, 5))


@pytest.fixture
def non_self_data():
    """Non-self data far from the self cluster."""
    rng = np.random.default_rng(99)
    return rng.uniform(0.7, 0.95, size=(20, 5))


@pytest.fixture
def trained_classifier(clustered_self_data):
    """Classifier trained on clustered self-data with permissive r_self."""
    config = NegSelConfig(num_detectors=10, r_self=0.3)
    clf = NegativeSelectionClassifier(config=config, class_label="SELF")
    np.random.seed(123)
    clf.fit(clustered_self_data, max_attempts=10000)
    return clf


class TestNegSelCoreGuarantee:
    """The fundamental guarantee: detectors never fire on self."""

    def test_detectors_never_fire_on_self(self, trained_classifier, clustered_self_data):
        """FALSIFICATION: If any self-sample is classified as non-self, the guarantee is broken."""
        for i, sample in enumerate(clustered_self_data):
            prediction = trained_classifier.predict_single(sample)
            assert prediction == 0.0, (
                f"Self-sample {i} was classified as non-self (prediction={prediction}). "
                f"This violates the NegSl-AIS R_q > R_self guarantee."
            )

    def test_self_anomaly_scores_are_zero(self, trained_classifier, clustered_self_data):
        """Self-data should have zero anomaly score (within R_self)."""
        for sample in clustered_self_data:
            score = trained_classifier.get_anomaly_score(sample)
            assert score == 0.0, f"Self-sample has non-zero anomaly score: {score}"


class TestNegSelDetection:
    """Detectors should fire on non-self data."""

    def test_detects_non_self(self, trained_classifier, non_self_data):
        """Non-self data far from the self cluster should be detected."""
        detected = sum(
            1 for sample in non_self_data
            if trained_classifier.predict_single(sample) == 1.0
        )
        detection_rate = detected / len(non_self_data)
        assert detection_rate > 0.5, (
            f"Detection rate {detection_rate:.2f} is too low. "
            f"Expected >50% detection of clearly non-self data."
        )

    def test_non_self_anomaly_scores_positive(self, trained_classifier, non_self_data):
        """Non-self data should have positive anomaly scores."""
        positive_scores = sum(
            1 for sample in non_self_data
            if trained_classifier.get_anomaly_score(sample) > 0
        )
        rate = positive_scores / len(non_self_data)
        assert rate > 0.5, f"Only {rate:.0%} of non-self samples have positive anomaly scores"


class TestNegSelDeterminism:
    """Same seed, same data -> same detectors."""

    def test_deterministic_given_seed(self, clustered_self_data):
        config = NegSelConfig(num_detectors=5, r_self=0.3)

        np.random.seed(777)
        clf1 = NegativeSelectionClassifier(config=config)
        clf1.fit(clustered_self_data)

        np.random.seed(777)
        clf2 = NegativeSelectionClassifier(config=config)
        clf2.fit(clustered_self_data)

        assert len(clf1.valid_detectors) == len(clf2.valid_detectors)
        for d1, d2 in zip(clf1.valid_detectors, clf2.valid_detectors):
            np.testing.assert_array_almost_equal(d1.center, d2.center)
            assert abs(d1.radius - d2.radius) < 1e-10


class TestNegSelConfig:
    """Test configuration and presets."""

    def test_preset_configs_exist(self):
        from immunos_mcp.algorithms.negsel import NEGSEL_PRESETS
        for key in ["LA", "HA", "LV", "HV", "LLM_HALLUCINATION", "GENERAL"]:
            assert key in NEGSEL_PRESETS

    def test_string_config_lookup(self):
        clf = NegativeSelectionClassifier(config="LA")
        assert clf.config.num_detectors == 15
        assert clf.config.r_self == 0.87

    def test_invalid_config_falls_back_to_general(self):
        clf = NegativeSelectionClassifier(config="NONEXISTENT")
        assert clf.config.num_detectors == 20  # GENERAL default

    def test_detector_radius_is_nonnegative(self, trained_classifier):
        for d in trained_classifier.valid_detectors:
            assert d.radius >= 0, f"Detector has negative radius: {d.radius}"


class TestMetrics:
    """Test evaluation metrics."""

    def test_mcc_perfect(self):
        assert calculate_mcc(10, 10, 0, 0) == 1.0

    def test_mcc_inverse(self):
        assert calculate_mcc(0, 0, 10, 10) == -1.0

    def test_mcc_random(self):
        result = calculate_mcc(5, 5, 5, 5)
        assert abs(result) < 0.01  # Near zero for random

    def test_kappa_perfect(self):
        result = calculate_kappa(50, 50, 0, 0)
        assert result == 1.0

    def test_kappa_zero_when_random(self):
        result = calculate_kappa(25, 25, 25, 25)
        assert abs(result) < 0.01
