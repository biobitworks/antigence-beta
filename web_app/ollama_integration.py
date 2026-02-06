"""
Ollama LLM Integration for Antigence Thymus Orchestrator
Uses local models for claim validation and reasoning
"""

import json
import requests
from typing import Dict, List, Optional


class OllamaOrchestrator:
    """
    Thymus Orchestrator using local Ollama LLM
    Coordinates antigent responses and provides final validation
    """

    def __init__(self, model="qwen2.5-coder:7b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m.get("name", "").startswith(self.model) for m in models)
        except:
            pass
        return False

    def orchestrate_validation(
        self,
        claim: str,
        bcell_verdict: str,
        bcell_confidence: float,
        nk_anomaly: bool,
        evidence_sentences: List[str] = None
    ) -> Dict:
        """
        Thymus orchestration: LLM reviews all antigent signals
        and provides final coordinated verdict

        Args:
            claim: Scientific claim to validate
            bcell_verdict: B Cell classification (SUPPORTS/CONTRADICTS/NOT ENOUGH INFO)
            bcell_confidence: B Cell confidence score
            nk_anomaly: NK Cell anomaly detection result
            evidence_sentences: Retrieved evidence from corpus

        Returns:
            dict with final_verdict, confidence_adjustment, reasoning
        """

        # Build prompt with immune system context
        prompt = self._build_orchestration_prompt(
            claim, bcell_verdict, bcell_confidence, nk_anomaly, evidence_sentences
        )

        try:
            # Call Ollama
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for consistent reasoning
                        "top_p": 0.9,
                        "num_predict": 300
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                llm_response = result.get("response", "")
                return self._parse_llm_response(llm_response, bcell_verdict, bcell_confidence)
            else:
                return self._fallback_response(bcell_verdict, bcell_confidence)

        except Exception as e:
            print(f"Ollama orchestration error: {e}")
            return self._fallback_response(bcell_verdict, bcell_confidence)

    def _build_orchestration_prompt(
        self,
        claim: str,
        bcell_verdict: str,
        bcell_confidence: float,
        nk_anomaly: bool,
        evidence_sentences: List[str] = None
    ) -> str:
        """Build immune-system-themed prompt for LLM"""

        prompt = f"""You are the Thymus Orchestrator in an artificial immune system for scientific claim validation.

CLAIM: "{claim}"

ANTIGENT SIGNALS:
- B Cell (Pattern Matcher): Verdict = {bcell_verdict}, Confidence = {bcell_confidence:.1%}
- NK Cell (Anomaly Detector): Anomaly Detected = {nk_anomaly}"""

        if evidence_sentences:
            prompt += f"\n- Evidence: {' '.join(evidence_sentences[:2])}"

        prompt += """

As the Thymus Orchestrator, coordinate these signals and provide:
1. FINAL_VERDICT: SUPPORTS | CONTRADICTS | NOT ENOUGH INFO | UNCERTAIN
2. CONFIDENCE_ADJUSTMENT: +X% or -X% (range: -20% to +20%)
3. REASONING: One sentence explaining your coordination decision

Format your response as:
VERDICT: [choice]
ADJUSTMENT: [+/-X%]
REASONING: [explanation]

Focus on:
- If NK Cell detected anomaly + B Cell says SUPPORTS → downgrade to UNCERTAIN
- If evidence contradicts B Cell → adjust verdict
- Maintain scientific rigor and conservative confidence

Response:"""

        return prompt

    def _parse_llm_response(
        self,
        llm_response: str,
        fallback_verdict: str,
        fallback_confidence: float
    ) -> Dict:
        """Parse LLM response into structured output"""

        verdict = fallback_verdict
        adjustment = 0.0
        reasoning = "Thymus orchestrator coordinated antigent signals."

        lines = llm_response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("VERDICT:"):
                verdict_text = line.split(":", 1)[1].strip().upper()
                if verdict_text in ["SUPPORTS", "CONTRADICTS", "NOT ENOUGH INFO", "UNCERTAIN"]:
                    verdict = verdict_text
            elif line.startswith("ADJUSTMENT:"):
                adj_text = line.split(":", 1)[1].strip().replace("%", "")
                try:
                    adjustment = float(adj_text) / 100.0
                    adjustment = max(-0.20, min(0.20, adjustment))  # Clamp to ±20%
                except:
                    adjustment = 0.0
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return {
            "final_verdict": verdict,
            "confidence_adjustment": adjustment,
            "llm_reasoning": reasoning,
            "llm_model": self.model
        }

    def _fallback_response(self, verdict: str, confidence: float) -> Dict:
        """Fallback if LLM unavailable"""
        return {
            "final_verdict": verdict,
            "confidence_adjustment": 0.0,
            "llm_reasoning": "LLM orchestrator unavailable, using antigent consensus.",
            "llm_model": None
        }


def test_ollama():
    """Test Ollama integration"""
    orchestrator = OllamaOrchestrator()

    print("Testing Ollama Orchestrator")
    print("-" * 60)
    print(f"Model: {orchestrator.model}")
    print(f"Available: {orchestrator.is_available()}")
    print()

    if orchestrator.is_available():
        result = orchestrator.orchestrate_validation(
            claim="Aspirin reduces the risk of cardiovascular disease.",
            bcell_verdict="NOT ENOUGH INFO",
            bcell_confidence=0.38,
            nk_anomaly=False,
            evidence_sentences=["Aspirin is commonly used for pain relief.", "Some studies suggest cardiovascular benefits."]
        )
        print("Orchestration Result:")
        print(json.dumps(result, indent=2))
    else:
        print("⚠️  Ollama not available. Start with: ollama serve")


if __name__ == "__main__":
    test_ollama()
