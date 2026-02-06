"""
Protocol definitions for IMMUNOS-MCP

Defines shared data structures and communication protocols
between immune system agents.
"""

from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum


class RecognitionStrategy(Enum):
    """Recognition strategies for classification"""
    SHA = "sha"  # Simple Highest Avidity
    RHA = "rha"  # Relative Highest Avidity
    CONSENSUS = "consensus"  # Multi-agent consensus
    HYBRID = "hybrid"  # Combination of strategies


class SignalType(Enum):
    """Signal types for dendritic cell processing"""
    PAMP = "pamp"  # Pathogen-Associated Molecular Pattern
    DANGER = "danger"  # Danger signal
    SAFE = "safe"  # Safe signal
    INFLAMMATORY = "inflammatory"  # Inflammatory signal


@dataclass
class RecognitionResult:
    """Result from pattern recognition"""
    predicted_class: Optional[str]
    confidence: float
    is_uncertain: bool = False
    avidity_scores: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    agents_involved: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "is_uncertain": self.is_uncertain,
            "avidity_scores": self.avidity_scores,
            "explanation": self.explanation,
            "agents_involved": self.agents_involved,
            "metadata": self.metadata
        }


@dataclass
class AnomalyResult:
    """Result from anomaly detection"""
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    detected_patterns: List[str] = field(default_factory=list)
    explanation: str = ""
    detector_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "detected_patterns": self.detected_patterns,
            "explanation": self.explanation,
            "detector_id": self.detector_id,
            "metadata": self.metadata
        }


@dataclass
class Signal:
    """Signal processed by dendritic cells"""
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    source: str
    evidence: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "source": self.source,
            "evidence": self.evidence,
            "timestamp": self.timestamp
        }


@dataclass
class ExtractedFeatures:
    """Features extracted by dendritic cells"""
    text_features: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    signals: List[Signal] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    danger_score: float = 0.0
    pamp_score: float = 0.0
    safe_score: float = 0.0

    def get_overall_signal(self) -> float:
        """Calculate overall signal strength"""
        danger_weight = 0.5
        pamp_weight = 0.3
        safe_weight = 0.2

        return (danger_weight * self.danger_score +
                pamp_weight * self.pamp_score -
                safe_weight * self.safe_score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text_features": self.text_features,
            "embedding": self.embedding,
            "signals": [s.to_dict() for s in self.signals],
            "context": self.context,
            "danger_score": self.danger_score,
            "pamp_score": self.pamp_score,
            "safe_score": self.safe_score
        }


@dataclass
class AgentRequest:
    """Request sent to an agent"""
    request_type: Literal["classify", "detect_anomaly", "extract_features", "retrieve_memory"]
    data: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "request_type": self.request_type,
            "data": self.data,
            "parameters": self.parameters,
            "request_id": self.request_id
        }


@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_name: str
    agent_type: str  # bcell, nk_cell, dendritic, memory, etc.
    result: Any
    success: bool = True
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "result": self.result,
            "success": self.success,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


@dataclass
class Pattern:
    """Learned pattern (like B cell antibody)"""
    pattern_id: str
    class_label: str
    example_data: Any
    embedding: Optional[List[float]] = None
    features: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    match_count: int = 0
    creation_time: Optional[float] = None
    last_updated: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "class_label": self.class_label,
            "example_data": self.example_data,
            "embedding": self.embedding,
            "features": self.features,
            "confidence": self.confidence,
            "match_count": self.match_count,
            "creation_time": self.creation_time,
            "last_updated": self.last_updated
        }


@dataclass
class Detector:
    """Detector for negative selection (NK cell)"""
    detector_id: str
    self_patterns: List[Pattern]  # Trained on "self" (normal) data
    threshold: float = 0.5
    creation_time: Optional[float] = None

    def matches_self(self, antigen: Any) -> bool:
        """Check if antigen matches self patterns (returns True if normal)"""
        # Implementation would check similarity to self_patterns
        return False  # Placeholder

    def is_non_self(self, antigen: Any) -> bool:
        """Check if antigen is non-self (anomaly)"""
        return not self.matches_self(antigen)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "detector_id": self.detector_id,
            "self_patterns": [p.to_dict() for p in self.self_patterns],
            "threshold": self.threshold,
            "creation_time": self.creation_time
        }


@dataclass
class Clone:
    """Clone population (group of B cells with similar patterns)"""
    clone_id: str
    class_label: str
    patterns: List[Pattern]
    size: int = 0
    avidity: float = 0.0
    concentration: float = 1.0

    def __post_init__(self):
        if self.size == 0:
            self.size = len(self.patterns)

    def calculate_avidity(self, affinity_scores: List[float]) -> float:
        """Calculate avidity using Immunos-81 formula"""
        import math
        if not affinity_scores:
            return 0.0
        total_affinity = sum(affinity_scores)
        concentration_factor = math.log(1 + self.size) * self.concentration
        return total_affinity * concentration_factor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "clone_id": self.clone_id,
            "class_label": self.class_label,
            "patterns": [p.to_dict() for p in self.patterns],
            "size": self.size,
            "avidity": self.avidity,
            "concentration": self.concentration
        }
