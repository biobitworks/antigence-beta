"""
Antigence Guardrails — Deterministic guardrails for probabilistic models.

Usage:
    from immunos_mcp.guardrails import GuardrailPipeline, GuardrailResult

    pipeline = GuardrailPipeline()
    result = pipeline.validate_output("LLM generated text here")
    if result.blocked:
        print(f"Blocked: {result.reason}")
"""

from .pipeline import GuardrailPipeline, GuardrailResult, GuardrailConfig

__all__ = ["GuardrailPipeline", "GuardrailResult", "GuardrailConfig"]
