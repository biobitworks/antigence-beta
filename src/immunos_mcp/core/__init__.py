"""Core modules for IMMUNOS-MCP"""

from .antigen import Antigen, AntigenBatch, DataType
from .affinity import AffinityCalculator, AffinityResult, DistanceMetric
from .protocols import (
    RecognitionResult,
    AnomalyResult,
    RecognitionStrategy,
    Signal,
    SignalType,
    ExtractedFeatures,
    Pattern,
    Detector,
    Clone,
    AgentRequest,
    AgentResponse,
)

__all__ = [
    "Antigen",
    "AntigenBatch",
    "DataType",
    "AffinityCalculator",
    "AffinityResult",
    "DistanceMetric",
    "RecognitionResult",
    "AnomalyResult",
    "RecognitionStrategy",
    "Signal",
    "SignalType",
    "ExtractedFeatures",
    "Pattern",
    "Detector",
    "Clone",
    "AgentRequest",
    "AgentResponse",
]
