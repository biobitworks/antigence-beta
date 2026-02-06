"""
Antigent LLM - Unified LLM caller for all immune cell roles.

Each antibody (trained ML model) can trigger its corresponding antigent (LLM)
when deeper analysis is needed.

Roles:
    - bcell: Evidence extraction, citation checks
    - nk: Risk summaries, anomaly explanations
    - tcell_security: Secure-code review
    - dendritic_summarizer: Summarize long artifacts
    - orchestrator: Coordinate all signals (Thymus)
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_roles import load_role_model_mapping, DEFAULT_ROLES


@dataclass
class AntigentResponse:
    """Response from an antigent (LLM) call."""
    role: str
    model: str
    content: str
    success: bool
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    timestamp: str = ""


class AntigentLLM:
    """
    Unified LLM caller for all antigent roles.

    Maps immune cell roles to specific Ollama models and provides
    role-specific prompting.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        events_path: Optional[str] = None,
    ):
        self.base_url = base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.default_model = default_model or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        self.events_path = events_path or os.environ.get(
            "ANTIGENCE_ANTIGENT_EVENTS_PATH",
            str(Path.home() / ".antigence" / "antigent_events.jsonl")
        )

        # Load role → model mapping
        self.role_mapping, self.mapping_meta = load_role_model_mapping()

        # Role-specific system prompts
        self.role_prompts = {
            "bcell": self._bcell_system_prompt(),
            "nk": self._nk_system_prompt(),
            "tcell_security": self._tcell_security_system_prompt(),
            "dendritic_summarizer": self._dendritic_summarizer_system_prompt(),
            "orchestrator": self._orchestrator_system_prompt(),
        }

    def get_model_for_role(self, role: str) -> str:
        """Get the configured model for a role."""
        return self.role_mapping.get(role, self.default_model)

    def call(
        self,
        role: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AntigentResponse:
        """
        Call an antigent (LLM) for a specific role.

        Args:
            role: The antigent role (bcell, nk, tcell_security, etc.)
            prompt: The user prompt
            context: Additional context (antibody results, etc.)
            temperature: LLM temperature
            max_tokens: Max response tokens

        Returns:
            AntigentResponse with the LLM output
        """
        import time
        start_time = time.time()

        model = self.get_model_for_role(role)
        system_prompt = self.role_prompts.get(role, "You are a helpful assistant.")

        # Build full prompt with context
        full_prompt = self._build_prompt(role, prompt, context)

        try:
            import requests

            response = requests.post(
                f"{self.base_url.rstrip('/')}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
                timeout=60
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code != 200:
                return AntigentResponse(
                    role=role,
                    model=model,
                    content="",
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    latency_ms=latency_ms,
                    timestamp=datetime.now().isoformat(),
                )

            data = response.json()
            content = data.get("response", "")

            result = AntigentResponse(
                role=role,
                model=model,
                content=content,
                success=True,
                tokens_used=data.get("eval_count", 0),
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
            )

            # Log the event
            self._log_event(role, prompt, context, result)

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return AntigentResponse(
                role=role,
                model=model,
                content="",
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
            )

    def _build_prompt(
        self,
        role: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the full prompt with context."""
        parts = []

        if context:
            parts.append("## Context from Antibody Analysis\n")
            for key, value in context.items():
                if isinstance(value, (dict, list)):
                    parts.append(f"**{key}**: {json.dumps(value, indent=2)}\n")
                else:
                    parts.append(f"**{key}**: {value}\n")
            parts.append("\n---\n\n")

        parts.append(prompt)
        return "".join(parts)

    def _log_event(
        self,
        role: str,
        prompt: str,
        context: Optional[Dict[str, Any]],
        result: AntigentResponse
    ) -> None:
        """Log antigent call to events file."""
        try:
            event = {
                "ts": datetime.now().isoformat(),
                "type": "antigent_call",
                "role": role,
                "model": result.model,
                "success": result.success,
                "tokens": result.tokens_used,
                "latency_ms": round(result.latency_ms, 2),
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "context_keys": list(context.keys()) if context else [],
                "error": result.error,
            }

            path = Path(self.events_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass  # Don't fail on logging errors

    # =========================================================================
    # ROLE-SPECIFIC SYSTEM PROMPTS
    # =========================================================================

    def _bcell_system_prompt(self) -> str:
        return """You are a B Cell Antigent - an AI specialized in evidence analysis and pattern recognition.

Your role in the immune response:
- Analyze claims and evidence for validity
- Extract key evidence sentences
- Verify citations and references
- Explain your verdict with specific evidence

You receive context from the B Cell antibody (ML pattern matcher) and provide deeper analysis.

Output format:
- Be concise and specific
- Quote evidence directly when possible
- Explain confidence levels
- Flag any inconsistencies"""

    def _nk_system_prompt(self) -> str:
        return """You are an NK Cell Antigent - an AI specialized in anomaly detection and risk assessment.

Your role in the immune response:
- Explain why something was flagged as anomalous
- Assess the risk level (LOW/MEDIUM/HIGH/CRITICAL)
- Identify what makes this "non-self" or unusual
- Suggest investigation steps

You receive context from the NK Cell antibody (negative selection detector) when an anomaly is detected.

Output format:
- Risk Level: [LOW|MEDIUM|HIGH|CRITICAL]
- Anomaly Type: [description]
- Why Flagged: [explanation]
- Recommended Action: [next steps]"""

    def _tcell_security_system_prompt(self) -> str:
        return """You are a T Cell Security Antigent - an AI specialized in secure code review.

Your role in the immune response:
- Identify security vulnerabilities in code
- Classify by CWE/OWASP categories
- Assess severity (LOW/MEDIUM/HIGH/CRITICAL)
- Provide specific remediation suggestions

You are the primary defense against malicious or vulnerable code.

Output format:
- Severity: [LOW|MEDIUM|HIGH|CRITICAL]
- Vulnerability Type: [CWE-XXX: Name]
- Location: [line/function if identifiable]
- Description: [what's wrong]
- Remediation: [how to fix]
- Safe Code Example: [corrected version]"""

    def _dendritic_summarizer_system_prompt(self) -> str:
        return """You are a Dendritic Summarizer Antigent - an AI specialized in processing and presenting information.

Your role in the immune response:
- Summarize long artifacts into key points
- Extract the most important signals
- Present information clearly for other immune cells
- Highlight danger signals or anomalies

Output format:
- Use bullet points
- Maximum 5-7 key points
- Lead with most important findings
- Flag any danger signals with [!]"""

    def _orchestrator_system_prompt(self) -> str:
        return """You are the Thymus Orchestrator - the central coordinator of the Antigence immune system.

Your role:
- Receive signals from all immune cells (B Cell, NK Cell, T Cell, Dendritic)
- Weigh conflicting signals
- Produce a final coordinated verdict
- Adjust confidence based on signal agreement

You are the final decision maker. Be decisive but explain your reasoning.

Output format:
- Final Verdict: [SAFE|DANGER|UNCERTAIN]
- Confidence: [0.0-1.0]
- Signal Summary: [which cells agreed/disagreed]
- Reasoning: [why this verdict]
- Recommended Action: [what should happen next]"""

    # =========================================================================
    # CONVENIENCE METHODS FOR EACH ROLE
    # =========================================================================

    def bcell_analyze(
        self,
        claim: str,
        antibody_result: Dict[str, Any],
    ) -> AntigentResponse:
        """B Cell antigent: Analyze a claim with evidence."""
        prompt = f"""Analyze this claim and the antibody's pattern match result:

CLAIM: {claim}

Provide detailed evidence analysis and explain the verdict."""

        return self.call("bcell", prompt, context=antibody_result)

    def nk_risk_summary(
        self,
        input_data: str,
        antibody_result: Dict[str, Any],
    ) -> AntigentResponse:
        """NK Cell antigent: Summarize risk for detected anomaly."""
        prompt = f"""An anomaly was detected by the NK Cell antibody. Analyze the risk:

INPUT: {input_data[:500]}...

Provide a risk assessment and recommended actions."""

        return self.call("nk", prompt, context=antibody_result)

    def tcell_security_review(
        self,
        code: str,
        quick_scan_result: Optional[Dict[str, Any]] = None,
    ) -> AntigentResponse:
        """T Cell antigent: Deep security review of code."""
        prompt = f"""Review this code for security vulnerabilities:

```
{code}
```

Identify all security issues, classify by CWE, and provide remediation."""

        return self.call("tcell_security", prompt, context=quick_scan_result)

    def dendritic_summarize(
        self,
        text: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> AntigentResponse:
        """Dendritic antigent: Summarize long text."""
        prompt = f"""Summarize this text into key bullet points:

{text}

Extract the most important signals and any danger indicators."""

        return self.call("dendritic_summarizer", prompt, context=features)

    def orchestrate(
        self,
        all_signals: Dict[str, Any],
    ) -> AntigentResponse:
        """Thymus orchestrator: Coordinate all immune signals."""
        prompt = """Coordinate all immune cell signals and produce a final verdict.

Weigh the evidence from each cell, resolve any conflicts, and provide
a definitive assessment with confidence level."""

        return self.call("orchestrator", prompt, context=all_signals)


# Singleton instance
_antigent_llm: Optional[AntigentLLM] = None


def get_antigent_llm() -> AntigentLLM:
    """Get the singleton AntigentLLM instance."""
    global _antigent_llm
    if _antigent_llm is None:
        _antigent_llm = AntigentLLM()
    return _antigent_llm
