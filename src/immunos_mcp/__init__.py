"""
IMMUNOS: Artificial Immune System Multi-Agent System

Standalone AIS implementation. MCP is optional for packaging or integration.
"""

__version__ = "0.1.0"

from .core.antigen import Antigen, AntigenBatch, DataType
from .core.affinity import AffinityCalculator, AffinityResult
from .core.protocols import (
    RecognitionResult,
    AnomalyResult,
    RecognitionStrategy,
    Signal,
    SignalType,
)

__all__ = [
    "Antigen",
    "AntigenBatch",
    "DataType",
    "AffinityCalculator",
    "AffinityResult",
    "RecognitionResult",
    "AnomalyResult",
    "RecognitionStrategy",
    "Signal",
    "SignalType",
]
