"""
LLM-Enhanced Immune System Agents for IMMUNOS-MCP

Wraps base agents with Ollama LLM capabilities for:
- Semantic code understanding
- Natural language explanations
- Improved anomaly detection
- Qualitative reasoning

Model assignments (16GB configuration):
- B Cell + Dendritic: qwen2.5-coder:7b (code analysis)
- NK Cell + T Cell + QML: deepseek-r1:14b (deep reasoning)
- Macrophage: qwen2.5:1.5b (fast triage)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import ollama
except ImportError:
    print("Warning: ollama package not installed. LLM features will be disabled.")
    print("Install with: pip install ollama")
    ollama = None

from immunos_mcp.agents.bcell_agent import BCellAgent
from immunos_mcp.core.antigen import Antigen
from immunos_mcp.core.protocols import AgentResponse, RecognitionResult

# ==============================================================================
# LLM Utility Functions
# ==============================================================================


def is_ollama_available() -> bool:
    """Check if Ollama is available."""
    if ollama is None:
        return False
    try:
        ollama.list()
        return True
    except:
        return False


def generate_with_ollama(model: str, prompt: str, max_tokens: int = 500) -> Optional[str]:
    """
    Generate text with Ollama model.

    Args:
        model: Model name (e.g., "qwen2.5-coder:7b")
        prompt: Input prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text or None if failed
    """
    if not is_ollama_available():
        return None

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "num_predict": max_tokens,
                "temperature": 0.7,
            },
        )
        return response.get("response", "").strip()
    except Exception as e:
        print(f"Warning: Ollama generation failed: {e}")
        return None


# ==============================================================================
# 1. LLM-Enhanced B Cell Agent (Pattern Matching)
# ==============================================================================


class LLMEnhancedBCellAgent(BCellAgent):
    """
    B Cell agent with LLM-powered semantic understanding.

    Model: qwen2.5-coder:7b (4.7GB)
    Purpose: Understand code semantics for better pattern matching
    """

    def __init__(
        self,
        agent_name: str = "llm_bcell_001",
        affinity_method: str = "hybrid",
        llm_model: str = "qwen2.5-coder:7b",
    ):
        super().__init__(agent_name, affinity_method)
        self.llm_model = llm_model
        self.llm_available = is_ollama_available()

    def calculate_semantic_similarity(self, code1: str, code2: str) -> float:
        """
        Calculate semantic similarity using LLM.

        Args:
            code1: First code snippet
            code2: Second code snippet

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not self.llm_available:
            return 0.5  # Fallback to neutral score

        prompt = f"""Compare these two code snippets for semantic similarity.
Rate from 0.0 (completely different) to 1.0 (identical purpose).
Output ONLY a single number, no explanation.

Code 1:
```
{code1[:500]}
```

Code 2:
```
{code2[:500]}
```

Similarity score:"""

        response = generate_with_ollama(self.llm_model, prompt, max_tokens=10)

        if response:
            try:
                # Extract number from response
                score = float(response.strip().split()[0])
                return max(0.0, min(1.0, score))
            except:
                pass

        return 0.5

    def explain_pattern_match(self, antigen: Antigen, result: RecognitionResult) -> str:
        """
        Generate natural language explanation of pattern match.

        Args:
            antigen: Test antigen
            result: Recognition result

        Returns:
            Human-readable explanation
        """
        if not self.llm_available:
            return result.explanation

        code = antigen.data[:1000]  # Limit context

        prompt = f"""Explain why this code was classified as "{result.predicted_class}".
Be concise (2-3 sentences). Focus on key patterns.

Code:
```
{code}
```

Classification: {result.predicted_class}
Confidence: {result.confidence:.2%}

Explanation:"""

        explanation = generate_with_ollama(self.llm_model, prompt, max_tokens=150)

        return explanation if explanation else result.explanation

    def recognize(
        self,
        antigen: Antigen,
        antigen_embedding: Optional[np.ndarray] = None,
        strategy: str = "sha",
        explain: bool = False,
    ) -> RecognitionResult:
        """
        Recognize with optional LLM explanation.

        Args:
            antigen: Antigen to classify
            antigen_embedding: Optional embedding
            strategy: Recognition strategy
            explain: Generate LLM explanation

        Returns:
            Recognition result with optional LLM explanation
        """
        # Base recognition
        result = super().recognize(antigen, antigen_embedding, strategy)

        # Add LLM explanation if requested
        if explain and self.llm_available:
            result.explanation = self.explain_pattern_match(antigen, result)
            result.metadata["llm_enhanced"] = True
            result.metadata["llm_model"] = self.llm_model

        return result


# ==============================================================================
# 2. LLM-Enhanced NK Cell Agent (Anomaly Detection)
# ==============================================================================


class LLMEnhancedNKCellAgent:
    """
    NK Cell agent with LLM-powered anomaly reasoning.

    Model: deepseek-r1:14b (8GB)
    Purpose: Deep reasoning about anomalies and threats
    """

    def __init__(self, agent_name: str = "llm_nkcell_001", llm_model: str = "deepseek-r1:14b"):
        self.agent_name = agent_name
        self.llm_model = llm_model
        self.llm_available = is_ollama_available()
        self.baseline_behaviors: Dict[str, Any] = {}

    def learn_baseline(self, normal_samples: List[str]):
        """
        Learn baseline behavior from normal samples.

        Args:
            normal_samples: List of normal code samples
        """
        print(f"[{self.agent_name}] Learning baseline from {len(normal_samples)} samples...")

        if not self.llm_available:
            self.baseline_behaviors = {"sample_count": len(normal_samples)}
            return

        # Use LLM to summarize normal behavior
        samples_text = "\n\n".join(
            [f"Sample {i+1}:\n{s[:300]}" for i, s in enumerate(normal_samples[:5])]
        )

        prompt = f"""Analyze these normal code samples and describe typical behavior patterns.
Focus on: API calls, file operations, network usage, data flow.
Be concise (3-4 bullet points).

{samples_text}

Normal behavior patterns:"""

        summary = generate_with_ollama(self.llm_model, prompt, max_tokens=200)

        self.baseline_behaviors = {
            "sample_count": len(normal_samples),
            "description": summary if summary else "Baseline learned from samples",
        }

        print(f"[{self.agent_name}] Baseline learned")

    def detect_anomaly(self, code: str, explain: bool = False) -> Tuple[bool, float, str]:
        """
        Detect if code is anomalous.

        Args:
            code: Code to analyze
            explain: Generate detailed explanation

        Returns:
            (is_anomalous, confidence, explanation)
        """
        if not self.llm_available:
            # Fallback: simple heuristics
            suspicious_keywords = ["eval", "exec", "subprocess", "os.system", "__import__"]
            found = [kw for kw in suspicious_keywords if kw in code]
            is_anomalous = len(found) > 0
            confidence = min(1.0, len(found) * 0.3)
            explanation = (
                f"Found suspicious keywords: {', '.join(found)}" if found else "No anomalies"
            )
            return is_anomalous, confidence, explanation

        # LLM-based anomaly detection
        baseline_desc = self.baseline_behaviors.get("description", "normal safe code")

        prompt = f"""Analyze this code for anomalies and potential threats.

Normal behavior: {baseline_desc}

Code to analyze:
```
{code[:1000]}
```

Is this code anomalous? Answer in this format:
ANOMALOUS: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASON: [brief explanation]"""

        response = generate_with_ollama(self.llm_model, prompt, max_tokens=200)

        if not response:
            return False, 0.0, "LLM analysis failed"

        # Parse response
        is_anomalous = "YES" in response.upper()
        confidence = 0.5

        # Try to extract confidence
        for line in response.split("\n"):
            if "CONFIDENCE" in line.upper():
                try:
                    confidence = float(line.split(":")[1].strip())
                except:
                    pass

        # Extract reason
        reason_lines = [l for l in response.split("\n") if "REASON" in l.upper()]
        explanation = reason_lines[0].split(":", 1)[1].strip() if reason_lines else response

        return is_anomalous, confidence, explanation

    def recognize(self, antigen: Antigen, explain: bool = False) -> RecognitionResult:
        """
        Recognize (detect anomalies) in antigen.

        Args:
            antigen: Code antigen
            explain: Generate detailed explanation

        Returns:
            Recognition result
        """
        start_time = time.time()

        is_anomalous, confidence, explanation = self.detect_anomaly(antigen.data, explain)

        result = RecognitionResult(
            predicted_class="anomalous" if is_anomalous else "normal",
            confidence=confidence,
            is_uncertain=confidence < 0.7,
            avidity_scores={"anomalous": confidence, "normal": 1.0 - confidence},
            explanation=explanation,
        )

        result.agents_involved = [self.agent_name]
        result.metadata["execution_time"] = time.time() - start_time
        result.metadata["llm_enhanced"] = self.llm_available
        result.metadata["llm_model"] = self.llm_model if self.llm_available else "none"

        return result


# ==============================================================================
# 3. LLM-Enhanced T Cell Agent (Coordinator)
# ==============================================================================


class LLMEnhancedTCellAgent:
    """
    T Cell coordinator with LLM-powered decision making.

    Model: deepseek-r1:14b (8GB)
    Purpose: Coordinate multiple agents and synthesize findings
    """

    def __init__(self, agent_name: str = "llm_tcell_001", llm_model: str = "deepseek-r1:14b"):
        self.agent_name = agent_name
        self.llm_model = llm_model
        self.llm_available = is_ollama_available()

    def coordinate(
        self, antigen: Antigen, agent_results: List[RecognitionResult], explain: bool = False
    ) -> RecognitionResult:
        """
        Coordinate multiple agent responses.

        Args:
            antigen: Test antigen
            agent_results: Results from other agents
            explain: Generate detailed coordination explanation

        Returns:
            Coordinated result
        """
        start_time = time.time()

        if not agent_results:
            return RecognitionResult(
                predicted_class="unknown",
                confidence=0.0,
                is_uncertain=True,
                explanation="No agent results to coordinate",
            )

        # Simple coordination without LLM
        votes: Dict[str, float] = {}
        for result in agent_results:
            if result.predicted_class:
                votes[result.predicted_class] = (
                    votes.get(result.predicted_class, 0) + result.confidence
                )

        predicted_class = max(votes, key=votes.get) if votes else "unknown"
        confidence = votes.get(predicted_class, 0) / len(agent_results)

        explanation = f"Consensus from {len(agent_results)} agents: {predicted_class}"

        # Enhance with LLM reasoning
        if explain and self.llm_available:
            agent_summary = "\n".join(
                [
                    f"- Agent: {r.predicted_class} (confidence: {r.confidence:.2%})"
                    for r in agent_results[:5]
                ]
            )

            code_snippet = antigen.data[:800]

            prompt = f"""Synthesize findings from multiple security agents analyzing this code.

Code:
```
{code_snippet}
```

Agent Findings:
{agent_summary}

Provide a coordinated assessment (2-3 sentences):"""

            llm_explanation = generate_with_ollama(self.llm_model, prompt, max_tokens=200)

            if llm_explanation:
                explanation = llm_explanation

        result = RecognitionResult(
            predicted_class=predicted_class,
            confidence=confidence,
            is_uncertain=confidence < 0.5,
            avidity_scores=votes,
            explanation=explanation,
        )

        result.agents_involved = [self.agent_name] + [
            r.agents_involved[0] for r in agent_results if r.agents_involved
        ]
        result.metadata["execution_time"] = time.time() - start_time
        result.metadata["agents_coordinated"] = len(agent_results)
        result.metadata["llm_enhanced"] = self.llm_available

        return result


# ==============================================================================
# 4. LLM-Enhanced Dendritic Cell Agent (Feature Extraction)
# ==============================================================================


class LLMEnhancedDendriticAgent:
    """
    Dendritic cell with LLM-powered feature extraction.

    Model: qwen2.5-coder:7b (4.7GB)
    Purpose: Extract semantic features from code
    """

    def __init__(self, agent_name: str = "llm_dendritic_001", llm_model: str = "qwen2.5-coder:7b"):
        self.agent_name = agent_name
        self.llm_model = llm_model
        self.llm_available = is_ollama_available()

    def extract_features(self, code: str) -> Dict[str, Any]:
        """
        Extract features from code using LLM.

        Args:
            code: Source code

        Returns:
            Feature dictionary
        """
        # Basic features (always available)
        features = {
            "length": len(code),
            "lines": code.count("\n") + 1,
            "has_imports": "import" in code,
            "has_functions": "def " in code,
        }

        if not self.llm_available:
            return features

        # LLM-enhanced feature extraction
        prompt = f"""Analyze this code and extract key features.

Code:
```
{code[:1000]}
```

Provide features in this format (one per line):
- API_CALLS: [list]
- FILE_OPS: [yes/no]
- NETWORK: [yes/no]
- RISK_LEVEL: [low/medium/high]"""

        response = generate_with_ollama(self.llm_model, prompt, max_tokens=150)

        if response:
            # Parse LLM response
            for line in response.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace("- ", "")
                    value = value.strip()
                    features[f"llm_{key}"] = value

        return features

    def process_antigen(self, antigen: Antigen) -> Antigen:
        """
        Process antigen and add extracted features.

        Args:
            antigen: Input antigen

        Returns:
            Antigen with enhanced features
        """
        features = self.extract_features(antigen.data)

        # Update antigen features
        if antigen.features:
            antigen.features.update(features)
        else:
            antigen.features = features

        return antigen


# ==============================================================================
# 5. LLM-Enhanced Macrophage Agent (Triage)
# ==============================================================================


class LLMEnhancedMacrophageAgent:
    """
    Macrophage with LLM-powered fast triage.

    Model: qwen2.5:1.5b (1.0GB)
    Purpose: Quick initial assessment
    """

    def __init__(self, agent_name: str = "llm_macrophage_001", llm_model: str = "qwen2.5:1.5b"):
        self.agent_name = agent_name
        self.llm_model = llm_model
        self.llm_available = is_ollama_available()

    def triage(self, code: str) -> Dict[str, Any]:
        """
        Quick triage of code sample.

        Args:
            code: Source code

        Returns:
            Triage result with priority and category
        """
        # Fast heuristics
        priority = "low"
        category = "safe"

        suspicious_patterns = {
            "eval": "high",
            "exec": "high",
            "__import__": "high",
            "subprocess": "medium",
            "os.system": "medium",
            "requests": "low",
            "urllib": "low",
        }

        max_priority = "low"
        for pattern, level in suspicious_patterns.items():
            if pattern in code:
                category = "suspicious"
                if level == "high":
                    max_priority = "high"
                elif level == "medium" and max_priority == "low":
                    max_priority = "medium"

        priority = max_priority

        result = {
            "priority": priority,
            "category": category,
            "requires_deep_analysis": priority in ["medium", "high"],
        }

        # Optional LLM enhancement (fast model)
        if self.llm_available and priority != "low":
            prompt = f"""Quick triage of this code. Is it safe or suspicious?
Answer: [SAFE/SUSPICIOUS] - [1 sentence reason]

Code:
{code[:500]}

Triage:"""

            response = generate_with_ollama(self.llm_model, prompt, max_tokens=50)

            if response:
                result["llm_triage"] = response.strip()

        return result


# ==============================================================================
# 6. QML Agent (Qualitative Model Learning)
# ==============================================================================


class LLMEnhancedQMLAgent:
    """
    QML agent with LLM-powered qualitative reasoning.

    Model: deepseek-r1:14b (8GB)
    Purpose: Learn qualitative behavioral models
    """

    def __init__(self, agent_name: str = "llm_qml_001", llm_model: str = "deepseek-r1:14b"):
        self.agent_name = agent_name
        self.llm_model = llm_model
        self.llm_available = is_ollama_available()

    def learn_behavior_model(self, code_samples: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Learn qualitative behavioral model.

        Args:
            code_samples: Training code samples
            labels: Labels (e.g., "safe", "malicious")

        Returns:
            Learned model description
        """
        if not self.llm_available:
            return {
                "model_type": "heuristic",
                "samples_count": len(code_samples),
                "classes": list(set(labels)),
            }

        # Sample a few examples per class
        examples_by_class: Dict[str, List[str]] = {}
        for sample, label in zip(code_samples, labels):
            if label not in examples_by_class:
                examples_by_class[label] = []
            if len(examples_by_class[label]) < 2:
                examples_by_class[label].append(sample[:300])

        # Build prompt
        examples_text = "\n\n".join(
            [
                f"Class: {label}\n" + "\n".join([f"Example: {ex}" for ex in examples])
                for label, examples in examples_by_class.items()
            ]
        )

        prompt = f"""Learn qualitative behavioral model from these code examples.
Describe patterns that distinguish each class (3-4 bullet points per class).

{examples_text}

Behavioral model:"""

        model_description = generate_with_ollama(self.llm_model, prompt, max_tokens=300)

        return {
            "model_type": "qualitative",
            "samples_count": len(code_samples),
            "classes": list(examples_by_class.keys()),
            "description": model_description if model_description else "Model learned from samples",
        }

    def recognize(self, antigen: Antigen, model: Dict[str, Any]) -> RecognitionResult:
        """
        Recognize using qualitative model.

        Args:
            antigen: Test antigen
            model: Learned model

        Returns:
            Recognition result
        """
        start_time = time.time()

        if not self.llm_available or "description" not in model:
            # Fallback
            return RecognitionResult(
                predicted_class="unknown",
                confidence=0.5,
                is_uncertain=True,
                explanation="QML model not available",
            )

        # Use LLM to classify based on learned model
        code = antigen.data[:1000]
        model_desc = model["description"]

        prompt = f"""Classify this code using the learned behavioral model.

Learned Model:
{model_desc}

Code to classify:
```
{code}
```

Classification: [class name] - [confidence 0.0-1.0] - [reason]"""

        response = generate_with_ollama(self.llm_model, prompt, max_tokens=150)

        if not response:
            return RecognitionResult(
                predicted_class="unknown",
                confidence=0.0,
                is_uncertain=True,
                explanation="Classification failed",
            )

        # Parse response
        parts = response.split("-")
        predicted_class = parts[0].strip() if len(parts) > 0 else "unknown"
        confidence = 0.5
        explanation = response

        if len(parts) > 1:
            try:
                confidence = float(parts[1].strip())
            except:
                pass

        result = RecognitionResult(
            predicted_class=predicted_class,
            confidence=confidence,
            is_uncertain=confidence < 0.7,
            explanation=explanation,
        )

        result.agents_involved = [self.agent_name]
        result.metadata["execution_time"] = time.time() - start_time
        result.metadata["llm_enhanced"] = True
        result.metadata["llm_model"] = self.llm_model

        return result
