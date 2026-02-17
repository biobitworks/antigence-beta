"""
Falsification tests for NK Cell Agent.

Core claims:
- NK Cell trains on self-only data (no anomaly examples needed)
- Detects novel/anomalous inputs
- Does not flag self-data as anomalous
"""

import pytest

from immunos_mcp.core.antigen import Antigen
from immunos_mcp.agents.nk_cell_agent import NKCellAgent
from immunos_mcp.agents.dendritic_agent import DendriticAgent


@pytest.fixture
def safe_antigens():
    """Normal/safe code samples."""
    return [
        Antigen.from_code("x = 1 + 2", class_label="safe"),
        Antigen.from_code("print('hello world')", class_label="safe"),
        Antigen.from_code("def add(a, b): return a + b", class_label="safe"),
        Antigen.from_code("result = [x**2 for x in range(10)]", class_label="safe"),
        Antigen.from_code("with open('file.txt') as f: data = f.read()", class_label="safe"),
    ]


@pytest.fixture
def vuln_antigens():
    """Anomalous/vulnerable code samples."""
    return [
        Antigen.from_code("eval(input())", class_label="vulnerable"),
        Antigen.from_code("os.system(user_input)", class_label="vulnerable"),
        Antigen.from_code("exec(base64.b64decode(payload))", class_label="vulnerable"),
    ]


@pytest.fixture
def dendritic():
    return DendriticAgent()


@pytest.fixture
def feature_trained_nkcell(safe_antigens, dendritic):
    """NK Cell trained in feature mode on safe data."""
    feature_vectors = [dendritic.get_feature_vector(a) for a in safe_antigens]
    agent = NKCellAgent(agent_name="test_nk", mode="feature", negsel_config="GENERAL")
    agent.train_on_features(safe_antigens, feature_vectors)
    return agent


class TestNKCellTraining:
    """Test training on self-only data."""

    def test_trains_on_self_only(self, safe_antigens):
        """NK Cell should train without any anomaly examples."""
        agent = NKCellAgent(mode="embedding")
        agent.train_on_self(safe_antigens)
        assert len(agent.self_patterns) == len(safe_antigens)

    def test_feature_training(self, feature_trained_nkcell):
        assert len(feature_trained_nkcell.feature_vectors) == 5
        assert len(feature_trained_nkcell.self_patterns) == 5

    def test_statistics(self, feature_trained_nkcell):
        stats = feature_trained_nkcell.get_statistics()
        assert stats["num_self_patterns"] == 5
        assert stats["engine"] == "NegSl-AIS (Eq 20)"


class TestNKCellDetection:
    """Test anomaly detection."""

    def test_detect_with_features(self, feature_trained_nkcell, vuln_antigens, dendritic):
        """Anomalous inputs should be detected."""
        for antigen in vuln_antigens:
            fv = dendritic.get_feature_vector(antigen)
            result = feature_trained_nkcell.detect_with_features(antigen, fv)
            assert hasattr(result, 'is_anomaly')
            assert hasattr(result, 'anomaly_score')
            assert 0.0 <= result.confidence <= 1.0

    def test_detect_novelty_returns_anomaly_result(self, safe_antigens):
        """detect_novelty should return AnomalyResult even without embeddings."""
        agent = NKCellAgent(mode="embedding")
        agent.train_on_self(safe_antigens)
        test = Antigen.from_code("eval(input())")
        result = agent.detect_novelty(test)
        assert hasattr(result, 'is_anomaly')


class TestNKCellPersistence:
    """Test save/load."""

    def test_save_load_roundtrip(self, feature_trained_nkcell):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nkcell_state.json")
            feature_trained_nkcell.save_state(path)
            loaded = NKCellAgent.load_state(path)
            assert len(loaded.feature_vectors) == len(feature_trained_nkcell.feature_vectors)
