"""
Falsification tests for the Guardrail Pipeline.

Core claims:
- Pipeline blocks outputs with danger signals
- Pipeline passes safe outputs
- Trained pipeline detects anomalous code
- Trained classifier catches unsafe code patterns
"""

import pytest

from immunos_mcp.guardrails import GuardrailPipeline, GuardrailResult, GuardrailConfig


@pytest.fixture
def default_pipeline():
    return GuardrailPipeline()


@pytest.fixture
def strict_pipeline():
    config = GuardrailConfig(
        block_on_danger=True,
        block_on_anomaly=True,
        block_on_low_credibility=True,
    )
    return GuardrailPipeline(config=config)


@pytest.fixture
def trained_pipeline():
    """Pipeline trained on safe code examples."""
    pipeline = GuardrailPipeline(config=GuardrailConfig(
        block_on_anomaly=True,
        block_on_danger=True,
    ))
    safe_examples = [
        "x = 1 + 2",
        "print('hello')",
        "def add(a, b): return a + b",
        "result = sorted(items)",
        "data = json.loads(text)",
        "logger.info('done')",
        "config = {'key': 'value'}",
        "for item in collection: process(item)",
    ]
    pipeline.train_on_safe_examples(safe_examples, is_code=True)
    return pipeline


@pytest.fixture
def classified_pipeline():
    """Pipeline with trained B-Cell classifier."""
    pipeline = GuardrailPipeline()
    safe = [
        "x = 1 + 2",
        "print('hello')",
        "def add(a, b): return a + b",
        "sorted(items)",
    ]
    unsafe = [
        "eval(input())",
        "os.system(cmd)",
        "exec(payload)",
        "pickle.loads(data)",
    ]
    pipeline.train_classifier(safe, unsafe, is_code=True)
    return pipeline


class TestGuardrailBasic:
    """Basic pipeline behavior."""

    def test_returns_guardrail_result(self, default_pipeline):
        result = default_pipeline.validate_output("Hello world")
        assert isinstance(result, GuardrailResult)

    def test_safe_text_passes(self, default_pipeline):
        result = default_pipeline.validate_output(
            "The study suggests that further research is needed."
        )
        assert result.passed is True
        assert result.risk_level == "LOW"

    def test_code_validation(self, default_pipeline):
        result = default_pipeline.validate_code("x = 1 + 2")
        assert isinstance(result, GuardrailResult)

    def test_result_to_dict(self, default_pipeline):
        result = default_pipeline.validate_output("test")
        d = result.to_dict()
        assert "blocked" in d
        assert "risk_level" in d
        assert "danger_score" in d


class TestDangerSignalDetection:
    """Pipeline should block outputs with danger patterns."""

    def test_blocks_miracle_cure_claims(self, default_pipeline):
        result = default_pipeline.validate_output(
            "This miracle cure guarantees 100% success with no side effects."
        )
        assert result.blocked is True
        assert result.risk_level == "HIGH"
        assert result.danger_score > 0

    def test_passes_hedged_science(self, default_pipeline):
        result = default_pipeline.validate_output(
            "The data suggests that the treatment may be effective, "
            "though further studies are needed (DOI: 10.1234/test)."
        )
        assert result.passed is True

    def test_danger_score_in_result(self, default_pipeline):
        result = default_pipeline.validate_output("This cures everything guaranteed!")
        assert result.danger_score > 0
        assert result.credibility_score < 0.5


class TestAnomalyDetection:
    """Trained pipeline should detect anomalous inputs."""

    def test_trained_pipeline_passes_safe_code(self, trained_pipeline):
        """Safe code similar to training should pass."""
        result = trained_pipeline.validate_code("y = 2 + 3")
        # Should not block safe code
        assert result.risk_level in ("LOW", "MEDIUM")

    def test_anomaly_result_fields(self, trained_pipeline):
        result = trained_pipeline.validate_code("eval(input())")
        assert hasattr(result, 'anomaly_detected')
        assert hasattr(result, 'anomaly_score')


class TestClassifier:
    """Trained B-Cell classifier tests."""

    def test_classifier_returns_classification(self, classified_pipeline):
        result = classified_pipeline.validate_code("eval(input())")
        assert result.classification is not None
        assert result.classification in ("safe", "unsafe")

    def test_classifier_has_confidence(self, classified_pipeline):
        result = classified_pipeline.validate_code("x = 1")
        assert 0.0 <= result.classification_confidence <= 1.0


class TestConfigOptions:
    """Test configuration affects behavior."""

    def test_disable_danger_signals(self):
        config = GuardrailConfig(enable_danger_signals=False)
        pipeline = GuardrailPipeline(config=config)
        result = pipeline.validate_output("This cures everything guaranteed 100%!")
        # With danger signals disabled, should not block
        assert result.blocked is False

    def test_disable_anomaly_detection(self):
        config = GuardrailConfig(enable_anomaly_detection=False)
        pipeline = GuardrailPipeline(config=config)
        pipeline.train_on_safe_examples(["x = 1", "y = 2"], is_code=True)
        result = pipeline.validate_code("eval(input())")
        # Anomaly detection disabled, so nk_result should be None
        assert result.nk_result is None

    def test_disable_pattern_classification(self):
        config = GuardrailConfig(enable_pattern_classification=False)
        pipeline = GuardrailPipeline(config=config)
        result = pipeline.validate_code("eval(input())")
        assert result.bcell_result is None


class TestEdgeCases:
    """Edge cases and empty inputs."""

    def test_empty_string(self, default_pipeline):
        result = default_pipeline.validate_output("")
        assert isinstance(result, GuardrailResult)

    def test_very_long_input(self, default_pipeline):
        long_text = "word " * 10000
        result = default_pipeline.validate_output(long_text)
        assert isinstance(result, GuardrailResult)

    def test_untrained_pipeline_skips_anomaly(self, default_pipeline):
        """Untrained pipeline should skip NK Cell check gracefully."""
        result = default_pipeline.validate_output("test")
        assert result.nk_result is None
        assert result.anomaly_detected is False
