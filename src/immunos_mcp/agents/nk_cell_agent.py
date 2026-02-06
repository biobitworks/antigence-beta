"""
NK Cell Agent for IMMUNOS-MCP

Anomaly detection agent using Negative Selection Algorithm.
Inspired by Natural Killer cells and T-cell negative selection in the thymus.
"""

from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
import random
import json
from pathlib import Path

from ..core.antigen import Antigen
from ..core.affinity import AffinityCalculator, DistanceMetric
from ..core.protocols import Detector, Pattern, AnomalyResult, AgentResponse
from ..algorithms.negsel import NegativeSelectionClassifier, NEGSEL_PRESETS, NegSelConfig


class NKCellAgent:
    """
    NK Cell Agent - Anomaly Detector via Negative Selection with NegSl-AIS Core.
    """

    def __init__(self, agent_name: str = "nk_cell_001",
                 detection_threshold: float = 0.5,
                 num_detectors: int = 100,
                 mode: str = "embedding",
                 negsel_config: str = "GENERAL"):
        """
        Initialize NK Cell agent with NegSl-AIS core.
        """
        self.agent_name = agent_name
        self.detection_threshold = detection_threshold
        self.num_detectors = num_detectors
        self.mode = mode  # "embedding" or "feature"
        self.self_patterns: List[Pattern] = []  # Normal (self) patterns
        self.detectors: List[Detector] = []  # Negative selection detectors
        self.affinity_calculator = AffinityCalculator(method="hybrid")
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # NegSl-AIS Core Engine (Eq 20)
        self.negsel_config_key = negsel_config
        config = NEGSEL_PRESETS.get(negsel_config, NEGSEL_PRESETS["GENERAL"])
        self.core_classifier = NegativeSelectionClassifier(config=config)
        
        # Feature-based mode storage
        self.feature_vectors: List[np.ndarray] = []  # Self feature vectors
        self.detector_vectors: List[np.ndarray] = []  # Detector feature vectors
        self.feature_dim: int = 20  # Dendritic feature dimension

    def train_on_self(self, normal_data: List[Antigen],
                      embeddings: Optional[List[np.ndarray]] = None):
        """
        Train on normal ("self") data.
        """
        print(f"[{self.agent_name}] Training on {len(normal_data)} normal (self) patterns...")

        # Store self patterns
        for i, antigen in enumerate(normal_data):
            pattern = Pattern(
                pattern_id=f"{self.agent_name}_self_{i}",
                class_label="self",  # All normal data is "self"
                example_data=antigen.data,
                embedding=embeddings[i].tolist() if embeddings and i < len(embeddings) else None,
                features=antigen.features,
                creation_time=time.time()
            )
            self.self_patterns.append(pattern)

            if embeddings and i < len(embeddings):
                self.embedding_cache[pattern.pattern_id] = embeddings[i]

        # Generate negative selection detectors
        self._generate_detectors()

        print(f"[{self.agent_name}] Generated {len(self.detectors)} detectors")

    def train_on_features(self, normal_data: List[Antigen],
                          feature_vectors: List[List[float]]):
        """
        Train on normal ("self") data using Dendritic feature vectors.
        """
        print(f"[{self.agent_name}] Training on {len(normal_data)} feature vectors (mode=feature)...")

        self.mode = "feature"
        self.feature_vectors = [np.array(fv) for fv in feature_vectors]

        # Infer feature dimension from training data
        if self.feature_vectors:
            self.feature_dim = len(self.feature_vectors[0])

        # Store self patterns for reference
        for i, antigen in enumerate(normal_data):
            pattern = Pattern(
                pattern_id=f"{self.agent_name}_self_{i}",
                class_label="self",
                example_data=antigen.data,
                embedding=feature_vectors[i] if i < len(feature_vectors) else None,
                features=antigen.features,
                creation_time=time.time()
            )
            self.self_patterns.append(pattern)

        # Generate feature-based detectors
        self._generate_feature_detectors()
        print(f"[{self.agent_name}] Generated {len(self.detector_vectors)} feature-based detectors")

    def _generate_feature_detectors(self):
        """
        Generate detectors in feature space using negative selection.
        """
        if not self.feature_vectors:
            return

        # Use Core Engine for generation if applicable
        self.core_classifier.fit(np.array(self.feature_vectors))
        self.detector_vectors = [d.center for d in self.core_classifier.valid_detectors]

    def detect_with_features(self, antigen: Antigen,
                             feature_vector: List[float]) -> AnomalyResult:
        """
        Detect anomaly using Dendritic feature vector and NegSl-AIS logic.
        """
        start_time = time.time()
        fv = np.array(feature_vector)
        
        # Ensure engine is primed
        if self.core_classifier.self_samples is None and self.feature_vectors:
            self.core_classifier.self_samples = np.array(self.feature_vectors)

        is_non_self = self.core_classifier.predict_single(fv)
        anomaly_score = float(self.core_classifier.get_anomaly_score(fv))
        
        # Legacy compatibility for similarity scores
        min_self_distance = float(self.core_classifier._get_nearest_self_distance(fv))
        max_distance = np.sqrt(self.feature_dim)
        self_similarity = 1.0 - (min_self_distance / max_distance)
        detector_similarity = anomaly_score # Proxy

        is_anomaly = (is_non_self == 1.0)
        detector_matches = ["negsel_core_detector"] if is_anomaly else []

        confidence = abs(self_similarity - detector_similarity)
        confidence = min(1.0, max(0.1, confidence))

        execution_time = time.time() - start_time

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            confidence=confidence,
            detected_patterns=detector_matches,
            explanation=self._generate_feature_explanation(is_anomaly, self_similarity, detector_similarity, detector_matches),
            detector_id=self.agent_name,
            metadata={
                "execution_time": execution_time,
                "mode": "feature",
                "num_self_patterns": len(self.feature_vectors),
                "num_detectors": len(self.detector_vectors),
                "self_similarity": self_similarity,
                "detector_similarity": detector_similarity,
                "detector_matches": len(detector_matches)
            }
        )

    def _generate_feature_explanation(self, is_anomaly: bool,
                                      self_sim: float,
                                      detector_sim: float,
                                      detector_matches: List[str]) -> str:
        """Generate explanation for feature-based detection."""
        if is_anomaly:
            return f"Anomaly detected (NegSl-AIS): Low similarity to Self patterns."
        return f"Normal pattern: High feature similarity to self ({self_sim:.3f})"

    def _generate_detectors(self):
        """
        Generate detectors using legacy negative selection (for embeddings).
        """
        if not self.self_patterns:
            return

        detectors_generated = 0
        attempts = 0
        max_attempts = self.num_detectors * 10 

        while detectors_generated < self.num_detectors and attempts < max_attempts:
            attempts += 1
            if self.embedding_cache:
                sample_embedding = random.choice(list(self.embedding_cache.values()))
                detector_embedding = sample_embedding + np.random.normal(0, 0.5, sample_embedding.shape)
                detector_embedding = detector_embedding / np.linalg.norm(detector_embedding)

                matches_self = False
                for pattern_id, self_embedding in self.embedding_cache.items():
                    similarity = 1 - DistanceMetric.cosine_distance(detector_embedding, self_embedding)
                    if similarity > self.detection_threshold:
                        matches_self = True
                        break

                if not matches_self:
                    detector = Detector(
                        detector_id=f"{self.agent_name}_detector_{detectors_generated}",
                        self_patterns=self.self_patterns,
                        threshold=self.detection_threshold,
                        creation_time=time.time()
                    )
                    self.detectors.append(detector)
                    self.embedding_cache[detector.detector_id] = detector_embedding
                    detectors_generated += 1

    def detect_novelty(self, antigen: Antigen,
                      antigen_embedding: Optional[np.ndarray] = None) -> AnomalyResult:
        """
        Detect if antigen is novel/anomalous.
        """
        if self.mode == "feature" and antigen.features:
            # Reconstruct vector for detect_with_features
            vec = [float(v) for v in antigen.features.values()] if isinstance(antigen.features, dict) else list(antigen.features)
            return self.detect_with_features(antigen, vec)
            
        start_time = time.time()
        max_self_similarity = 0.0
        for pattern in self.self_patterns:
            pattern_embedding = self.embedding_cache.get(pattern.pattern_id)
            affinity = self.affinity_calculator.calculate(
                antigen.data,
                pattern.example_data,
                embeddings1=antigen_embedding,
                embeddings2=pattern_embedding
            )
            if affinity.score > max_self_similarity:
                max_self_similarity = affinity.score

        is_anomaly = (max_self_similarity < self.detection_threshold)
        anomaly_score = 1.0 - max_self_similarity
        confidence = min(1.0, max(0.1, anomaly_score))

        execution_time = time.time() - start_time

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            confidence=confidence,
            detector_id=self.agent_name,
            metadata={"execution_time": execution_time}
        )

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "num_self_patterns": len(self.self_patterns),
            "num_detectors": len(self.detectors),
            "engine": "NegSl-AIS (Eq 20)"
        }

    def to_agent_response(self, result: AnomalyResult, success: bool = True,
                         execution_time: float = 0.0) -> AgentResponse:
        return AgentResponse(
            agent_name=self.agent_name,
            agent_type="nk_cell",
            result=result.to_dict(),
            success=success,
            execution_time=execution_time,
            metadata=result.metadata
        )

    def save_state(self, path: str) -> None:
        state = {
            "agent_name": self.agent_name,
            "negsel_config": self.negsel_config_key if hasattr(self, 'negsel_config_key') else "GENERAL",
            "self_patterns": [pattern.to_dict() for pattern in self.self_patterns],
            "feature_vectors": [fv.tolist() for fv in self.feature_vectors],
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load_state(cls, path: str) -> "NKCellAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            agent_name=data.get("agent_name", "nk_cell_001"),
            negsel_config=data.get("negsel_config", "GENERAL"),
        )
        agent.feature_vectors = [np.array(fv) for fv in data.get("feature_vectors", [])]
        return agent
