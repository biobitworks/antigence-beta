"""
Code Security Scanner Datasets

Contains curated examples of safe and vulnerable code patterns
for training the immune system agents.
"""

from .safe_patterns import SAFE_PATTERNS
from .vulnerable_patterns import VULNERABLE_PATTERNS

__all__ = ["SAFE_PATTERNS", "VULNERABLE_PATTERNS"]
