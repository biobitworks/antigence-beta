"""
Immune Response — Three-Level Response System
==============================================
Maps continuous binding affinity scores to actionable responses.
Inspired by biological immune costimulation thresholds.

IGNORE: Input is self — no action needed
REVIEW: Input is uncertain — flag for human review
REJECT: Input is non-self — block/alert
"""

from dataclasses import dataclass
from enum import Enum


class ImmuneResponse(Enum):
    """Three-level immune response based on binding affinity."""
    IGNORE = "ignore"   # binding < low_threshold — self, no review
    REVIEW = "review"   # low_threshold <= binding < high_threshold — flag for human
    REJECT = "reject"   # binding >= high_threshold — block/alert


@dataclass
class ResponseThresholds:
    """Per-domain thresholds for the three-level response."""
    low: float = 0.3    # Below this = IGNORE (self)
    high: float = 0.7   # Above this = REJECT (non-self)

    def classify(self, binding_affinity: float) -> ImmuneResponse:
        """Classify a binding affinity score into an immune response."""
        if binding_affinity < self.low:
            return ImmuneResponse.IGNORE
        elif binding_affinity < self.high:
            return ImmuneResponse.REVIEW
        else:
            return ImmuneResponse.REJECT


# Domain-specific defaults
CITATION_THRESHOLDS = ResponseThresholds(low=0.3, high=0.7)
ANALYSIS_THRESHOLDS = ResponseThresholds(low=0.3, high=0.7)
SECURITY_THRESHOLDS = ResponseThresholds(low=0.2, high=0.5)  # More sensitive for security
