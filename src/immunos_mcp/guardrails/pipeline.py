"""
GuardrailPipeline — The core integration layer between Antigence AIS and LLM outputs.

Provides deterministic validation of probabilistic model outputs using:
- Dendritic Agent: Deterministic feature extraction (20-dim vector)
- NegSl-AIS: Negative selection for anomaly detection (zero FPR guarantee)
- B-Cell: Pattern classification (safe vs vulnerable)
- Crypto verification: Agent state integrity

This is the "deterministic guardrails for probabilistic models" promise.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


from ..agents.dendritic_agent import DendriticAgent
from ..agents.nk_cell_agent import NKCellAgent
from ..agents.bcell_agent import BCellAgent
from ..core.antigen import Antigen


@dataclass
class GuardrailConfig:
    """Configuration for the guardrail pipeline."""
    # Detection thresholds
    danger_signal_threshold: float = 0.5
    anomaly_threshold: float = 0.5
    confidence_threshold: float = 0.3

    # Which checks to run
    enable_danger_signals: bool = True
    enable_anomaly_detection: bool = True
    enable_pattern_classification: bool = True

    # Behavior
    block_on_danger: bool = True
    block_on_anomaly: bool = False  # Conservative default: warn, don't block
    block_on_low_credibility: bool = False

    # Credibility floor
    min_credibility: float = 0.3


@dataclass
class GuardrailResult:
    """Result from the guardrail pipeline."""
    blocked: bool
    passed: bool
    reason: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"

    # Detailed scores
    danger_score: float = 0.0
    credibility_score: float = 0.5
    anomaly_detected: bool = False
    anomaly_score: float = 0.0
    classification: Optional[str] = None
    classification_confidence: float = 0.0

    # Signal details
    signals: Dict[str, Any] = field(default_factory=dict)

    # Raw agent outputs
    dendritic_features: Optional[Dict[str, Any]] = None
    nk_result: Optional[Dict[str, Any]] = None
    bcell_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocked": self.blocked,
            "passed": self.passed,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "danger_score": self.danger_score,
            "credibility_score": self.credibility_score,
            "anomaly_detected": self.anomaly_detected,
            "anomaly_score": self.anomaly_score,
            "classification": self.classification,
            "classification_confidence": self.classification_confidence,
            "signals": self.signals,
        }


class GuardrailPipeline:
    """
    Deterministic guardrail pipeline for LLM output validation.

    Usage:
        # Basic usage
        pipeline = GuardrailPipeline()
        result = pipeline.validate_output("LLM response text")

        # Code validation
        result = pipeline.validate_code("eval(user_input)")

        # Custom config
        config = GuardrailConfig(block_on_danger=True, block_on_anomaly=True)
        pipeline = GuardrailPipeline(config=config)

        # With trained agents
        pipeline = GuardrailPipeline()
        pipeline.train_on_safe_examples(safe_texts)
        result = pipeline.validate_output(llm_response)
    """

    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self.dendritic = DendriticAgent(agent_name="guardrail_dendritic")
        self.nk_cell: Optional[NKCellAgent] = None
        self.bcell: Optional[BCellAgent] = None
        self._trained = False

    def train_on_safe_examples(
        self,
        safe_texts: List[str],
        is_code: bool = False,
    ) -> None:
        """
        Train the guardrail on examples of safe/normal output.

        After training, the NK Cell will flag outputs that differ
        significantly from the training distribution.

        Args:
            safe_texts: List of known-safe text or code examples
            is_code: Whether the examples are code (vs prose)
        """
        if is_code:
            antigens = [Antigen.from_code(t, class_label="safe") for t in safe_texts]
        else:
            antigens = [Antigen.from_text(t, class_label="safe") for t in safe_texts]

        feature_vectors = [self.dendritic.get_feature_vector(a) for a in antigens]

        self.nk_cell = NKCellAgent(
            agent_name="guardrail_nk",
            mode="feature",
            negsel_config="GENERAL",
        )
        self.nk_cell.train_on_features(antigens, feature_vectors)
        self._trained = True

    def train_classifier(
        self,
        safe_texts: List[str],
        unsafe_texts: List[str],
        is_code: bool = False,
    ) -> None:
        """
        Train the B-Cell classifier on labeled examples.

        Args:
            safe_texts: Known-safe examples
            unsafe_texts: Known-unsafe examples
            is_code: Whether examples are code
        """
        factory = Antigen.from_code if is_code else Antigen.from_text
        safe = [factory(t, class_label="safe") for t in safe_texts]
        unsafe = [factory(t, class_label="unsafe") for t in unsafe_texts]

        self.bcell = BCellAgent(
            agent_name="guardrail_bcell",
            affinity_method="traditional",
        )
        self.bcell.train(safe + unsafe)

    def validate_output(self, text: str) -> GuardrailResult:
        """
        Validate an LLM text output through the guardrail pipeline.

        Runs: Dendritic features -> Danger signals -> NK Cell -> B-Cell

        Args:
            text: The LLM output to validate

        Returns:
            GuardrailResult with pass/block decision and detailed scores
        """
        antigen = Antigen.from_text(text)
        return self._run_pipeline(antigen)

    def validate_code(self, code: str, language: str = "python") -> GuardrailResult:
        """
        Validate LLM-generated code through the guardrail pipeline.

        Args:
            code: The generated code to validate
            language: Programming language

        Returns:
            GuardrailResult with pass/block decision
        """
        antigen = Antigen.from_code(code, language=language)
        return self._run_pipeline(antigen)

    def _run_pipeline(self, antigen: Antigen) -> GuardrailResult:
        """Run the full guardrail pipeline on an antigen."""
        # Step 1: Dendritic feature extraction (always runs, deterministic)
        features = self.dendritic.extract_features(antigen)
        feature_vector = self.dendritic.get_feature_vector(antigen)
        signal_classification = self.dendritic.classify_signals(features)

        danger_score = features.get("pamp_score", 0.0)
        credibility = features.get("source_credibility", 0.5)

        blocked = False
        reasons = []
        risk_level = "LOW"

        # Step 2: Danger signal check
        if self.config.enable_danger_signals:
            if danger_score >= self.config.danger_signal_threshold:
                risk_level = "HIGH"
                reasons.append(f"danger_signals({danger_score:.2f})")
                if self.config.block_on_danger:
                    blocked = True

            if credibility < self.config.min_credibility:
                if risk_level != "HIGH":
                    risk_level = "MEDIUM"
                reasons.append(f"low_credibility({credibility:.2f})")
                if self.config.block_on_low_credibility:
                    blocked = True

        # Step 3: NK Cell anomaly detection (if trained)
        nk_result_dict = None
        anomaly_detected = False
        anomaly_score = 0.0

        if self.config.enable_anomaly_detection and self.nk_cell and self._trained:
            nk_result = self.nk_cell.detect_with_features(antigen, feature_vector)
            anomaly_detected = nk_result.is_anomaly
            anomaly_score = nk_result.anomaly_score
            nk_result_dict = nk_result.to_dict()

            if anomaly_detected:
                if risk_level == "LOW":
                    risk_level = "MEDIUM"
                reasons.append(f"anomaly({anomaly_score:.2f})")
                if self.config.block_on_anomaly:
                    blocked = True

        # Step 4: B-Cell classification (if trained)
        bcell_result_dict = None
        classification = None
        classification_confidence = 0.0

        if self.config.enable_pattern_classification and self.bcell:
            result = self.bcell.recognize(antigen)
            classification = result.predicted_class
            classification_confidence = result.confidence
            bcell_result_dict = result.to_dict()

            if classification == "unsafe" and classification_confidence > self.config.confidence_threshold:
                risk_level = "HIGH"
                reasons.append(f"classified_unsafe({classification_confidence:.2f})")
                blocked = True

        # Compose result
        if not reasons:
            reason = "All checks passed"
        else:
            reason = "; ".join(reasons)

        return GuardrailResult(
            blocked=blocked,
            passed=not blocked,
            reason=reason,
            risk_level=risk_level,
            danger_score=danger_score,
            credibility_score=credibility,
            anomaly_detected=anomaly_detected,
            anomaly_score=anomaly_score,
            classification=classification,
            classification_confidence=classification_confidence,
            signals=signal_classification,
            dendritic_features=features,
            nk_result=nk_result_dict,
            bcell_result=bcell_result_dict,
        )
