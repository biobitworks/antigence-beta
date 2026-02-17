"""
Falsification tests for B Cell Agent.

Core claims:
- B Cell can learn patterns and recognize new antigens
- Save/load preserves classification behavior
- Empty training is handled gracefully
"""

import json
import tempfile
from pathlib import Path

import pytest

from immunos_mcp.core.antigen import Antigen
from immunos_mcp.agents.bcell_agent import BCellAgent


@pytest.fixture
def training_data():
    """Labeled training antigens with two classes."""
    safe = [
        Antigen.from_code("x = 1 + 2", class_label="safe"),
        Antigen.from_code("print('hello')", class_label="safe"),
        Antigen.from_code("def add(a, b): return a + b", class_label="safe"),
    ]
    vuln = [
        Antigen.from_code("eval(input())", class_label="vulnerable"),
        Antigen.from_code("os.system(cmd)", class_label="vulnerable"),
        Antigen.from_code("exec(user_data)", class_label="vulnerable"),
    ]
    return safe + vuln


@pytest.fixture
def trained_agent(training_data):
    """B Cell agent trained on safe+vulnerable code."""
    agent = BCellAgent(agent_name="test_bcell", affinity_method="traditional")
    agent.train(training_data)
    return agent


class TestBCellTraining:
    """Test training lifecycle."""

    def test_train_creates_patterns(self, trained_agent, training_data):
        assert len(trained_agent.patterns) == len(training_data)

    def test_train_creates_clones_per_class(self, trained_agent):
        assert "safe" in trained_agent.clones
        assert "vulnerable" in trained_agent.clones
        assert trained_agent.clones["safe"].size == 3
        assert trained_agent.clones["vulnerable"].size == 3

    def test_train_ignores_unlabeled(self):
        agent = BCellAgent(affinity_method="traditional")
        data = [
            Antigen.from_code("x = 1", class_label="safe"),
            Antigen.from_code("y = 2", class_label=None),  # No label
        ]
        agent.train(data)
        assert len(agent.patterns) == 1

    def test_add_pattern_incremental(self, trained_agent):
        initial = len(trained_agent.patterns)
        trained_agent.add_pattern(Antigen.from_code("new code", class_label="safe"))
        assert len(trained_agent.patterns) == initial + 1
        assert trained_agent.clones["safe"].size == 4

    def test_add_pattern_new_class(self, trained_agent):
        trained_agent.add_pattern(Antigen.from_code("malware()", class_label="malicious"))
        assert "malicious" in trained_agent.clones

    def test_add_unlabeled_raises(self, trained_agent):
        with pytest.raises(ValueError):
            trained_agent.add_pattern(Antigen.from_code("no label"))


class TestBCellRecognition:
    """Test recognition/classification."""

    def test_recognize_returns_result(self, trained_agent):
        antigen = Antigen.from_code("eval(x)")
        result = trained_agent.recognize(antigen, strategy="sha")
        assert result.predicted_class in ("safe", "vulnerable")
        assert 0.0 <= result.confidence <= 1.0

    def test_recognize_rha_strategy(self, trained_agent):
        antigen = Antigen.from_code("print('ok')")
        result = trained_agent.recognize(antigen, strategy="rha")
        assert result.predicted_class is not None

    def test_empty_agent_returns_uncertain(self):
        agent = BCellAgent(affinity_method="traditional")
        antigen = Antigen.from_code("test")
        result = agent.recognize(antigen)
        assert result.is_uncertain

    def test_statistics(self, trained_agent):
        stats = trained_agent.get_statistics()
        assert stats["num_patterns"] == 6
        assert stats["num_clones"] == 2
        assert "safe" in stats["classes"]


class TestBCellPersistence:
    """Test save/load state."""

    def test_save_load_roundtrip(self, trained_agent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "bcell_state.json")
            trained_agent.save_state(path)

            loaded = BCellAgent.load_state(path)
            assert len(loaded.patterns) == len(trained_agent.patterns)
            assert set(loaded.clones.keys()) == set(trained_agent.clones.keys())

    def test_saved_file_is_valid_json(self, trained_agent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "bcell_state.json")
            trained_agent.save_state(path)
            data = json.loads(Path(path).read_text())
            assert "agent_name" in data
            assert "patterns" in data

    def test_loaded_agent_recognizes(self, trained_agent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "bcell_state.json")
            trained_agent.save_state(path)
            loaded = BCellAgent.load_state(path)

            antigen = Antigen.from_code("eval(x)")
            result = loaded.recognize(antigen)
            assert result.predicted_class is not None
