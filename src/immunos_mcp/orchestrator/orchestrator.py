"""
IMMUNOS Orchestrator

Coordinates multi-agent analysis for standalone IMMUNOS runs.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import argparse
import time
from pathlib import Path
import json
import os
import urllib.request
import numpy as np

from ..agents.bcell_agent import BCellAgent
from ..agents.nk_cell_agent import NKCellAgent
from ..agents.nk_cell_enhanced import EnhancedNKCellAgent
from ..agents.dendritic_agent import DendriticAgent
from ..agents.memory_agent import MemoryAgent
from ..core.antigen import Antigen
from ..embeddings.simple_text_embedder import SimpleTextEmbedder


@dataclass
class OrchestratorResult:
    """Aggregated result from IMMUNOS agents."""
    classification: Optional[str]
    confidence: float
    anomaly: bool
    bcell_confidence: float
    nk_confidence: float
    features: Dict[str, Any]
    memory_hit: bool
    signals: Dict[str, float]
    agents: List[str]
    model_roles: Dict[str, str]
    metadata: Dict[str, Any]


class ImmunosOrchestrator:
    """Run a multi-agent analysis workflow."""

    def __init__(self,
                 bcell_state: Optional[str] = None,
                 nk_state: Optional[str] = None):
        self.bcell = BCellAgent.load_state(bcell_state) if bcell_state else BCellAgent()
        self.nk_cell = NKCellAgent.load_state(nk_state) if nk_state else NKCellAgent()
        self.dendritic = DendriticAgent()
        self.memory = MemoryAgent()
        self.model_roles = self._load_model_roles()
        self.agent_routes = self._load_agent_routes()
        self.agent_cache: Dict[str, Dict[str, Any]] = {
            "default": {
                "bcell": self.bcell,
                "nk": self.nk_cell,
                "nk_enhanced": None,
                "embedding": None,
            }
        }
        self.embedder_cache = {
            "simple_text": SimpleTextEmbedder(),
        }

    def analyze(self, antigen: Antigen) -> OrchestratorResult:
        """Run B cell and NK cell analysis and aggregate."""
        start_time = time.time()

        features = self.dendritic.extract_features(antigen)
        signals = self._derive_signals(features)

        domain = self._resolve_domain(antigen)
        bcell_agent, nk_agent, nk_enhanced, embedder_key = self._get_agents_for_domain(domain)
        embedding = self._resolve_embedding(antigen, embedder_key)

        if embedding is not None:
            bcell_result = bcell_agent.recognize(antigen, antigen_embedding=embedding)
        else:
            bcell_result = bcell_agent.recognize(antigen)

        if nk_enhanced:
            nk_result = nk_enhanced.detect_novelty(antigen, antigen_embedding=embedding)
            nk_agent_name = nk_enhanced.agent_name
            nk_agent_type = "nk_cell_enhanced"
        else:
            nk_result = nk_agent.detect_novelty(antigen, antigen_embedding=embedding)
            nk_agent_name = nk_agent.agent_name
            nk_agent_type = "nk_cell"

        classification = bcell_result.predicted_class
        anomaly = nk_result.is_anomaly

        # Check Memory for similar past cases (few-shot learning)
        memory_key = antigen.identifier or str(hash(antigen.get_text_content()))
        previous = self.memory.retrieve(memory_key)
        memory_hit = previous is not None

        few_shot_examples = []
        if embedding is not None:
            emb_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            few_shot_examples = self.memory.get_few_shot_examples(emb_list, k=3)

        # Store classification for future few-shot learning
        from ..agents.memory_agent import generate_hash
        self.memory.store_classification(
            antigen_hash=generate_hash(antigen.get_text_content()),
            antigen_text=antigen.get_text_content()[:500],
            class_label=classification or "unknown",
            confidence=bcell_result.confidence,
            embedding=embedding.tolist() if hasattr(embedding, 'tolist') else (list(embedding) if embedding is not None else None),
            features=features
        )

        # Weighted aggregation with dendritic signals.
        base_confidence = (bcell_result.confidence + nk_result.confidence) / 2.0
        confidence = self._calibrate_confidence(base_confidence, signals, anomaly)

        metadata = {
            "execution_time": time.time() - start_time,
            "bcell": bcell_result.to_dict(),
            "nk_cell": nk_result.to_dict(),
            "signals": signals,
            "model_roles": self.model_roles,
            "domain": domain,
            "embedding": embedder_key,
            "nk_agent_type": nk_agent_type,
            "few_shot_examples": few_shot_examples,
            "memory_stats": self.memory.get_statistics(),
        }

        return OrchestratorResult(
            classification=classification,
            confidence=confidence,
            anomaly=anomaly,
            bcell_confidence=bcell_result.confidence,
            nk_confidence=nk_result.confidence,
            features=features,
            memory_hit=memory_hit,
            signals=signals,
            agents=[bcell_agent.agent_name, nk_agent_name, self.dendritic.agent_name, self.memory.agent_name],
            model_roles=self.model_roles,
            metadata=metadata,
        )

    def _derive_signals(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Create signal weights from Dendritic features."""
        # Use new Dendritic signal classification
        signal_class = self.dendritic.classify_signals(features)
        signal_type = signal_class.get("signal_type", "NEUTRAL")

        # Extract key features
        pamp_score = features.get("pamp_score", 0.0)
        danger_count = features.get("danger_signal_count", 0)
        credibility = features.get("source_credibility", 0.5)

        # Map to signal weights
        danger = pamp_score + (danger_count * 0.2)
        pamp = pamp_score
        safe = credibility if signal_type == "SAFE" else 0.0

        return {
            "danger": min(1.0, danger),
            "pamp": pamp,
            "safe": safe,
            "signal_type": signal_type,
            "credibility": credibility
        }

    def _calibrate_confidence(self,
                              base: float,
                              signals: Dict[str, float],
                              anomaly: bool) -> float:
        """Adjust confidence based on dendritic signals."""
        danger = signals.get("danger", 0.0)
        safe = signals.get("safe", 0.0)

        if anomaly:
            adjusted = base + (0.2 * danger) - (0.1 * safe)
        else:
            adjusted = base + (0.1 * safe) - (0.1 * danger)

        return max(0.1, min(1.0, adjusted))

    def _load_model_roles(self) -> Dict[str, str]:
        """Load model-role mapping from config without external dependencies."""
        config_path = Path(__file__).resolve().parents[1] / "config" / "model_roles.yaml"
        if not config_path.exists():
            return {}

        roles: Dict[str, str] = {}
        for line in config_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            roles[key.strip()] = value.strip()
        return roles

    def _load_agent_routes(self) -> Dict[str, Dict[str, str]]:
        """Load domain routing configuration."""
        config_path = Path(__file__).resolve().parents[1] / "config" / "agent_routes.json"
        if not config_path.exists():
            return {}
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _resolve_domain(self, antigen: Antigen) -> Optional[str]:
        return antigen.metadata.get("domain") or antigen.metadata.get("route")

    def _get_agents_for_domain(self, domain: Optional[str]) -> Tuple[BCellAgent, NKCellAgent, Optional[EnhancedNKCellAgent], Optional[str]]:
        if not domain or domain not in self.agent_routes:
            cached = self.agent_cache["default"]
            return cached["bcell"], cached["nk"], cached["nk_enhanced"], cached["embedding"]

        if domain in self.agent_cache:
            cached = self.agent_cache[domain]
            return cached["bcell"], cached["nk"], cached["nk_enhanced"], cached["embedding"]

        route = self.agent_routes[domain]
        bcell_agent = self.bcell
        nk_agent = self.nk_cell
        nk_enhanced = None

        bcell_state = route.get("bcell_state")
        if bcell_state and Path(bcell_state).exists():
            bcell_agent = BCellAgent.load_state(bcell_state)

        nk_enhanced_state = route.get("nk_enhanced_state")
        nk_state = route.get("nk_state")
        if nk_enhanced_state and Path(nk_enhanced_state).exists():
            nk_enhanced = EnhancedNKCellAgent.load_state(nk_enhanced_state)
        elif nk_state and Path(nk_state).exists():
            nk_agent = NKCellAgent.load_state(nk_state)

        embedding = route.get("embedding")
        self.agent_cache[domain] = {
            "bcell": bcell_agent,
            "nk": nk_agent,
            "nk_enhanced": nk_enhanced,
            "embedding": embedding,
        }

        return bcell_agent, nk_agent, nk_enhanced, embedding

    def _resolve_embedding(self, antigen: Antigen, embedder_key: Optional[str]) -> Optional[np.ndarray]:
        if not embedder_key:
            return None

        if embedder_key == "simple_text":
            embedder = self.embedder_cache["simple_text"]
            return embedder.embed(antigen.get_text_content())

        if embedder_key == "features":
            if isinstance(antigen.features, dict) and "features" in antigen.features:
                return np.array(antigen.features["features"], dtype=float)
            if isinstance(antigen.data, dict) and "features" in antigen.data:
                return np.array(antigen.data["features"], dtype=float)
            if isinstance(antigen.data, list):
                return np.array(antigen.data, dtype=float)

        return None

    def emit_dashboard_event(self,
                             result: OrchestratorResult,
                             domain: Optional[str],
                             dashboard_url: str) -> None:
        """Send a detection event to the dashboard API."""
        if not dashboard_url:
            return

        if result.confidence < 0.4:
            outcome = "uncertain"
        elif result.anomaly:
            outcome = "danger" if result.confidence >= 0.8 else "non_self"
        else:
            outcome = "self"

        payload = {
            "domain": domain or "unknown",
            "result": outcome,
            "confidence": result.confidence,
            "matched_detectors": [],
            "danger_signal": result.signals.get("danger", 0.0),
        }

        url = dashboard_url.rstrip("/") + "/api/events/detection"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                response.read()
        except Exception:
            return


def main() -> None:
    """CLI entrypoint for a minimal orchestrator run."""
    parser = argparse.ArgumentParser(description="IMMUNOS Orchestrator")
    parser.add_argument("--clear-memory", action="store_true", help="Clear memory store and exit")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--bcell-state", type=str, help="Path to saved B Cell state")
    parser.add_argument("--nk-state", type=str, help="Path to saved NK state")
    parser.add_argument("--domain", type=str, help="Route to a domain-specific agent set")
    parser.add_argument("--emit-dashboard", action="store_true", help="Emit result to dashboard API")
    parser.add_argument("--dashboard-url", type=str, default=None, help="Dashboard base URL")
    args = parser.parse_args()

    orchestrator = ImmunosOrchestrator(
        bcell_state=args.bcell_state,
        nk_state=args.nk_state,
    )

    if args.clear_memory:
        orchestrator.memory.clear()
        print("Memory cleared")
        return

    text = args.text or "Example input for IMMUNOS analysis"
    metadata = {"domain": args.domain} if args.domain else None
    antigen = Antigen.from_text(text, metadata=metadata)
    result = orchestrator.analyze(antigen)
    dashboard_url = args.dashboard_url or os.getenv("IMMUNOS_DASHBOARD_URL")
    if args.emit_dashboard or dashboard_url:
        orchestrator.emit_dashboard_event(result, args.domain, dashboard_url or "http://localhost:5001")
    print(result)


if __name__ == "__main__":
    main()
