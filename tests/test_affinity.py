"""
Falsification tests for AffinityCalculator.

Core claims:
- Affinity scores are bounded in [0, 1]
- Embedding affinity is symmetric: affinity(a,b) == affinity(b,a)
- Self-affinity is maximal: affinity(a,a) == 1.0
"""

import numpy as np
import pytest

from immunos_mcp.core.affinity import AffinityCalculator, DistanceMetric


@pytest.fixture
def calc_traditional():
    return AffinityCalculator(method="traditional")


@pytest.fixture
def calc_embedding():
    return AffinityCalculator(method="embedding")


@pytest.fixture
def calc_hybrid():
    return AffinityCalculator(method="hybrid", embedding_weight=0.7)


class TestAffinityBounds:
    """Affinity scores must be in [0, 1]."""

    def test_numeric_affinity_bounded(self, calc_traditional):
        pairs = [(0, 100), (1, 1), (-5, 5), (0.001, 999999), (0, 0)]
        for a, b in pairs:
            result = calc_traditional.calculate(a, b)
            assert 0.0 <= result.score <= 1.0, (
                f"affinity({a}, {b}) = {result.score} is out of bounds"
            )

    def test_string_affinity_bounded(self, calc_traditional):
        pairs = [("hello", "world"), ("", "test"), ("abc", "abc"), ("x", "y")]
        for a, b in pairs:
            result = calc_traditional.calculate(a, b)
            assert 0.0 <= result.score <= 1.0, (
                f"affinity('{a}', '{b}') = {result.score} is out of bounds"
            )

    def test_embedding_affinity_bounded(self, calc_embedding):
        rng = np.random.default_rng(42)
        for _ in range(50):
            e1 = rng.standard_normal(128)
            e2 = rng.standard_normal(128)
            result = calc_embedding.calculate("a", "b", embeddings1=e1, embeddings2=e2)
            assert 0.0 <= result.score <= 1.0, (
                f"Embedding affinity {result.score} is out of bounds"
            )

    def test_hybrid_affinity_bounded(self, calc_hybrid):
        rng = np.random.default_rng(42)
        e1 = rng.standard_normal(64)
        e2 = rng.standard_normal(64)
        result = calc_hybrid.calculate("hello", "world", embeddings1=e1, embeddings2=e2)
        assert 0.0 <= result.score <= 1.0


class TestAffinitySymmetry:
    """Embedding affinity must be symmetric."""

    def test_embedding_symmetry(self, calc_embedding):
        rng = np.random.default_rng(42)
        for _ in range(20):
            e1 = rng.standard_normal(64)
            e2 = rng.standard_normal(64)
            r1 = calc_embedding.calculate("a", "b", embeddings1=e1, embeddings2=e2)
            r2 = calc_embedding.calculate("b", "a", embeddings1=e2, embeddings2=e1)
            assert abs(r1.score - r2.score) < 1e-10, (
                f"Asymmetric: affinity(a,b)={r1.score} != affinity(b,a)={r2.score}"
            )

    def test_string_symmetry(self, calc_traditional):
        pairs = [("hello", "world"), ("abc", "xyz"), ("test", "testing")]
        for a, b in pairs:
            r1 = calc_traditional.calculate(a, b)
            r2 = calc_traditional.calculate(b, a)
            assert abs(r1.score - r2.score) < 1e-10, (
                f"Asymmetric: affinity('{a}','{b}')={r1.score} != affinity('{b}','{a}')={r2.score}"
            )


class TestSelfAffinity:
    """Self-affinity should be maximal."""

    def test_identical_strings(self, calc_traditional):
        result = calc_traditional.calculate("hello world", "hello world")
        assert result.score == 1.0

    def test_identical_numbers(self, calc_traditional):
        result = calc_traditional.calculate(42, 42)
        assert result.score == 1.0

    def test_identical_embeddings(self, calc_embedding):
        e = np.array([1.0, 2.0, 3.0])
        result = calc_embedding.calculate("a", "a", embeddings1=e, embeddings2=e)
        assert abs(result.score - 1.0) < 1e-10

    def test_zero_vector_returns_zero(self, calc_embedding):
        z = np.zeros(64)
        e = np.ones(64)
        result = calc_embedding.calculate("a", "b", embeddings1=z, embeddings2=e)
        assert result.score == 0.0


class TestDistanceMetric:
    """Test distance metric helper class."""

    def test_euclidean_self_is_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert DistanceMetric.euclidean(v, v) == 0.0

    def test_cosine_distance_self_is_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(DistanceMetric.cosine_distance(v, v)) < 1e-10

    def test_cosine_distance_bounded(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            v1 = rng.standard_normal(32)
            v2 = rng.standard_normal(32)
            d = DistanceMetric.cosine_distance(v1, v2)
            assert 0.0 <= d <= 1.0, f"Cosine distance {d} out of [0,1]"


class TestAvidity:
    """Test avidity calculation."""

    def test_avidity_empty_returns_zero(self):
        calc = AffinityCalculator()
        assert calc.calculate_avidity([], clone_size=5) == 0.0

    def test_avidity_increases_with_clone_size(self):
        calc = AffinityCalculator()
        affinities = [0.8, 0.9, 0.7]
        avidity_small = calc.calculate_avidity(affinities, clone_size=1)
        avidity_large = calc.calculate_avidity(affinities, clone_size=10)
        assert avidity_large > avidity_small
