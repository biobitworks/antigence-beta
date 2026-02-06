"""
Enhanced NK Cell Agent for IMMUNOS-MCP

Implements NegSl-AIS methodology for improved anomaly detection:
- Adaptive threshold calculation (min_distance, mean_distance, percentile)
- Per-class detector generation (one-vs-rest strategy)
- Enhanced validation rule: d(detector, nearest_self) > τ
- Quality metrics tracking
"""

from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
import random
from enum import Enum
import json
from pathlib import Path

from ..core.antigen import Antigen
from ..core.affinity import DistanceMetric
from ..core.protocols import Detector, Pattern, AnomalyResult, AgentResponse


class ThresholdMethod(Enum):
    """Methods for calculating adaptive thresholds"""
    MIN_DISTANCE = "min_distance"  # Half of minimum distance between self patterns
    MEAN_DISTANCE = "mean_distance"  # Mean distance between self patterns
    PERCENTILE = "percentile"  # 25th percentile of distances


class ClassDetectorSet:
    """Detectors for a specific class (one-vs-rest)"""

    def __init__(self, class_label: str, num_detectors: int = 20):
        self.class_label = class_label
        self.num_detectors = num_detectors
        self.detectors: List[np.ndarray] = []
        self.quality_scores: List[float] = []
        self.threshold: float = 0.5

    def add_detector(self, detector: np.ndarray, quality_score: float = 1.0):
        """Add detector with quality score"""
        self.detectors.append(detector)
        self.quality_scores.append(quality_score)

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            "class_label": self.class_label,
            "num_detectors": len(self.detectors),
            "threshold": self.threshold,
            "avg_quality": np.mean(self.quality_scores) if self.quality_scores else 0.0,
            "min_quality": np.min(self.quality_scores) if self.quality_scores else 0.0,
            "max_quality": np.max(self.quality_scores) if self.quality_scores else 0.0
        }


class EnhancedNKCellAgent:
    """
    Enhanced NK Cell Agent with NegSl-AIS methodology

    Improvements over base NK Cell:
    - Adaptive threshold calculation based on self-pattern distribution
    - Per-class detector generation (one-vs-rest strategy)
    - Enhanced validation rule: d(detector, nearest_self) > τ
    - Quality metrics for detector effectiveness
    """

    def __init__(self,
                 agent_name: str = "nk_cell_enhanced_001",
                 threshold_method: str = "min_distance",
                 detectors_per_class: int = 20,
                 max_generation_attempts: int = 1000):
        """
        Initialize Enhanced NK Cell agent.

        Args:
            agent_name: Unique identifier
            threshold_method: "min_distance", "mean_distance", or "percentile"
            detectors_per_class: Number of detectors to generate per class (15-25 recommended)
            max_generation_attempts: Maximum attempts to generate each detector
        """
        self.agent_name = agent_name
        self.threshold_method = ThresholdMethod(threshold_method)
        self.detectors_per_class = detectors_per_class
        self.max_generation_attempts = max_generation_attempts

        # Store self patterns by class
        self.self_patterns_by_class: Dict[str, List[Pattern]] = {}
        self.self_embeddings_by_class: Dict[str, List[np.ndarray]] = {}

        # Per-class detector sets
        self.detector_sets: Dict[str, ClassDetectorSet] = {}

        # Global threshold (can be overridden per class)
        self.global_threshold: float = 0.5

        # Statistics
        self.training_stats: Dict[str, Any] = {}

    def train_on_self(self,
                      normal_data: List[Antigen],
                      embeddings: Optional[List[np.ndarray]] = None,
                      class_labels: Optional[List[str]] = None):
        """
        Train on normal ("self") data with optional class labels.

        If class_labels provided, generates per-class detectors (one-vs-rest).
        Otherwise, treats all data as single "self" class.

        Args:
            normal_data: List of normal (non-anomalous) antigens
            embeddings: Optional embeddings for normal data
            class_labels: Optional class labels for one-vs-rest strategy
        """
        print(f"[{self.agent_name}] Training on {len(normal_data)} normal (self) patterns...")
        start_time = time.time()

        # Organize self patterns by class
        if class_labels:
            for i, (antigen, label) in enumerate(zip(normal_data, class_labels)):
                if label not in self.self_patterns_by_class:
                    self.self_patterns_by_class[label] = []
                    self.self_embeddings_by_class[label] = []

                pattern = Pattern(
                    pattern_id=f"{self.agent_name}_self_{label}_{i}",
                    class_label=label,
                    example_data=antigen.data,
                    embedding=embeddings[i].tolist() if embeddings and i < len(embeddings) else None,
                    features=antigen.features,
                    creation_time=time.time()
                )
                self.self_patterns_by_class[label].append(pattern)

                if embeddings and i < len(embeddings):
                    self.self_embeddings_by_class[label].append(embeddings[i])
        else:
            # Single class "self"
            self.self_patterns_by_class["self"] = []
            self.self_embeddings_by_class["self"] = []

            for i, antigen in enumerate(normal_data):
                pattern = Pattern(
                    pattern_id=f"{self.agent_name}_self_{i}",
                    class_label="self",
                    example_data=antigen.data,
                    embedding=embeddings[i].tolist() if embeddings and i < len(embeddings) else None,
                    features=antigen.features,
                    creation_time=time.time()
                )
                self.self_patterns_by_class["self"].append(pattern)

                if embeddings and i < len(embeddings):
                    self.self_embeddings_by_class["self"].append(embeddings[i])

        # Calculate adaptive thresholds and generate detectors per class
        for class_label, self_embeddings in self.self_embeddings_by_class.items():
            if len(self_embeddings) == 0:
                print(f"[{self.agent_name}] Warning: No embeddings for class '{class_label}'")
                continue

            # Calculate adaptive threshold
            threshold = self._calculate_adaptive_threshold(self_embeddings)

            # Generate detectors for this class
            detector_set = self._generate_class_detectors(
                class_label,
                self_embeddings,
                threshold
            )
            self.detector_sets[class_label] = detector_set

            print(f"[{self.agent_name}] Class '{class_label}': threshold={threshold:.4f}, "
                  f"detectors={len(detector_set.detectors)}")

        # Calculate global threshold (average of class thresholds)
        if self.detector_sets:
            self.global_threshold = np.mean([ds.threshold for ds in self.detector_sets.values()])

        training_time = time.time() - start_time
        self.training_stats = {
            "training_time": training_time,
            "num_classes": len(self.self_patterns_by_class),
            "total_self_patterns": sum(len(patterns) for patterns in self.self_patterns_by_class.values()),
            "total_detectors": sum(len(ds.detectors) for ds in self.detector_sets.values()),
            "global_threshold": self.global_threshold,
            "class_thresholds": {label: ds.threshold for label, ds in self.detector_sets.items()}
        }

        print(f"[{self.agent_name}] Training complete: {self.training_stats['total_detectors']} "
              f"detectors across {self.training_stats['num_classes']} classes "
              f"({training_time:.2f}s)")

    def _calculate_adaptive_threshold(self, self_embeddings: List[np.ndarray]) -> float:
        """
        Calculate adaptive threshold based on self-pattern distribution.

        Args:
            self_embeddings: Embeddings of self patterns

        Returns:
            Adaptive threshold value
        """
        if len(self_embeddings) < 2:
            return 0.5  # Default if insufficient data

        # Calculate pairwise distances
        distances = []
        for i in range(len(self_embeddings)):
            for j in range(i + 1, len(self_embeddings)):
                dist = DistanceMetric.cosine_distance(self_embeddings[i], self_embeddings[j])
                distances.append(dist)

        if not distances:
            return 0.5

        distances = np.array(distances)

        # Apply threshold method
        if self.threshold_method == ThresholdMethod.MIN_DISTANCE:
            # Half of minimum distance (NegSl-AIS method)
            threshold = np.min(distances) / 2.0
        elif self.threshold_method == ThresholdMethod.MEAN_DISTANCE:
            # Mean distance
            threshold = np.mean(distances)
        elif self.threshold_method == ThresholdMethod.PERCENTILE:
            # 25th percentile
            threshold = np.percentile(distances, 25)
        else:
            threshold = 0.5

        # Clamp to reasonable range
        threshold = max(0.1, min(0.9, threshold))

        return threshold

    def _generate_class_detectors(self,
                                  class_label: str,
                                  self_embeddings: List[np.ndarray],
                                  threshold: float) -> ClassDetectorSet:
        """
        Generate detectors for a specific class using enhanced validation.

        Implements NegSl-AIS validation rule:
        d(detector, nearest_self) > τ

        Args:
            class_label: Class label for this detector set
            self_embeddings: Embeddings of self patterns
            threshold: Adaptive threshold

        Returns:
            ClassDetectorSet with generated detectors
        """
        detector_set = ClassDetectorSet(class_label, self.detectors_per_class)
        detector_set.threshold = threshold

        if len(self_embeddings) == 0:
            return detector_set

        # Get embedding dimension
        embedding_dim = self_embeddings[0].shape[0]

        detectors_generated = 0
        total_attempts = 0

        while detectors_generated < self.detectors_per_class:
            if total_attempts >= self.max_generation_attempts * self.detectors_per_class:
                print(f"[{self.agent_name}] Warning: Reached max attempts for class '{class_label}'. "
                      f"Generated {detectors_generated}/{self.detectors_per_class} detectors")
                break

            total_attempts += 1

            # Generate random candidate detector
            # Strategy: Random point with normal perturbation from random self pattern
            base_pattern = random.choice(self_embeddings)
            candidate = base_pattern + np.random.normal(0, 0.5, embedding_dim)

            # Normalize
            norm = np.linalg.norm(candidate)
            if norm > 0:
                candidate = candidate / norm

            # Enhanced validation: d(detector, nearest_self) > threshold
            min_distance_to_self = float('inf')
            for self_embedding in self_embeddings:
                distance = DistanceMetric.cosine_distance(candidate, self_embedding)
                min_distance_to_self = min(min_distance_to_self, distance)

            # NegSl-AIS rule: distance to nearest self must exceed threshold
            if min_distance_to_self > threshold:
                # Calculate quality score (higher distance = better quality)
                quality_score = min_distance_to_self
                detector_set.add_detector(candidate, quality_score)
                detectors_generated += 1

        return detector_set

    def detect_novelty(self,
                      antigen: Antigen,
                      antigen_embedding: Optional[np.ndarray] = None,
                      target_class: Optional[str] = None) -> AnomalyResult:
        """
        Detect if antigen is novel/anomalous using enhanced negative selection.

        Args:
            antigen: Antigen to test
            antigen_embedding: Optional embedding
            target_class: Optional target class for one-vs-rest detection

        Returns:
            AnomalyResult with detection outcome
        """
        start_time = time.time()

        if antigen_embedding is None:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                explanation="No embedding provided",
                detector_id=self.agent_name
            )

        # Determine which detector sets to use
        if target_class and target_class in self.detector_sets:
            detector_sets_to_check = {target_class: self.detector_sets[target_class]}
        else:
            detector_sets_to_check = self.detector_sets

        # Check against each detector set
        max_anomaly_score = 0.0
        best_matching_class = None
        all_detector_matches = []

        for class_label, detector_set in detector_sets_to_check.items():
            # Check similarity to self patterns
            self_embeddings = self.self_embeddings_by_class.get(class_label, [])
            max_self_similarity = 0.0

            for self_embedding in self_embeddings:
                distance = DistanceMetric.cosine_distance(antigen_embedding, self_embedding)
                similarity = 1.0 - distance
                max_self_similarity = max(max_self_similarity, similarity)

            # Check detector matches
            detector_matches = 0
            max_detector_distance = 0.0

            for i, detector in enumerate(detector_set.detectors):
                distance = DistanceMetric.cosine_distance(antigen_embedding, detector)

                # Anomaly if distance to detector is small (detector matches)
                if distance < detector_set.threshold:
                    detector_matches += 1
                    quality = detector_set.quality_scores[i]
                    max_detector_distance = max(max_detector_distance, quality)

            # Calculate anomaly score for this class
            # High score if: low self-similarity OR high detector matches
            class_anomaly_score = max(
                1.0 - max_self_similarity,
                detector_matches / len(detector_set.detectors) if detector_set.detectors else 0.0
            )

            if class_anomaly_score > max_anomaly_score:
                max_anomaly_score = class_anomaly_score
                best_matching_class = class_label
                all_detector_matches = [(class_label, detector_matches)]

        # Decision: anomaly if score exceeds threshold
        is_anomaly = max_anomaly_score > 0.5

        # Confidence based on how clear the decision is
        confidence = abs(max_anomaly_score - 0.5) * 2.0  # Scale to 0-1
        confidence = min(1.0, max(0.1, confidence))

        explanation = self._generate_explanation(
            is_anomaly,
            max_anomaly_score,
            best_matching_class,
            all_detector_matches
        )

        execution_time = time.time() - start_time

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=max_anomaly_score,
            confidence=confidence,
            detected_patterns=[match[0] for match in all_detector_matches],
            explanation=explanation,
            detector_id=self.agent_name,
            metadata={
                "execution_time": execution_time,
                "best_matching_class": best_matching_class,
                "num_detector_sets": len(detector_sets_to_check),
                "total_detectors": sum(len(ds.detectors) for ds in detector_sets_to_check.values()),
                "threshold_method": self.threshold_method.value
            }
        )

    def _generate_explanation(self,
                            is_anomaly: bool,
                            anomaly_score: float,
                            best_class: Optional[str],
                            detector_matches: List[Tuple[str, int]]) -> str:
        """Generate human-readable explanation"""
        if is_anomaly:
            if best_class:
                matches = dict(detector_matches).get(best_class, 0)
                return (f"Anomaly detected (score: {anomaly_score:.3f}). "
                       f"Best match: class '{best_class}' with {matches} detector activations")
            else:
                return f"Anomaly detected (score: {anomaly_score:.3f})"
        else:
            return f"Normal pattern (score: {anomaly_score:.3f})"

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        detector_stats = {
            label: ds.get_statistics()
            for label, ds in self.detector_sets.items()
        }

        return {
            "agent_name": self.agent_name,
            "threshold_method": self.threshold_method.value,
            "detectors_per_class": self.detectors_per_class,
            "global_threshold": self.global_threshold,
            "num_classes": len(self.detector_sets),
            "detector_sets": detector_stats,
            "training_stats": self.training_stats
        }

    def to_agent_response(self,
                         result: AnomalyResult,
                         success: bool = True,
                         execution_time: float = 0.0) -> AgentResponse:
        """Convert anomaly result to agent response"""
        return AgentResponse(
            agent_name=self.agent_name,
            agent_type="nk_cell_enhanced",
            result=result.to_dict(),
            success=success,
            execution_time=execution_time,
            metadata=result.metadata
        )

    def save_state(self, path: str) -> None:
        """Persist enhanced NK state to disk."""
        state = {
            "agent_name": self.agent_name,
            "threshold_method": self.threshold_method.value,
            "detectors_per_class": self.detectors_per_class,
            "max_generation_attempts": self.max_generation_attempts,
            "global_threshold": self.global_threshold,
            "training_stats": self.training_stats,
            "self_patterns_by_class": {
                label: [pattern.to_dict() for pattern in patterns]
                for label, patterns in self.self_patterns_by_class.items()
            },
            "self_embeddings_by_class": {
                label: [embedding.tolist() for embedding in embeddings]
                for label, embeddings in self.self_embeddings_by_class.items()
            },
            "detector_sets": {
                label: {
                    "threshold": detector_set.threshold,
                    "detectors": [detector.tolist() for detector in detector_set.detectors],
                    "quality_scores": detector_set.quality_scores,
                }
                for label, detector_set in self.detector_sets.items()
            },
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load_state(cls, path: str) -> "EnhancedNKCellAgent":
        """Load enhanced NK state from disk."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            agent_name=data.get("agent_name", "nk_cell_enhanced_001"),
            threshold_method=data.get("threshold_method", "min_distance"),
            detectors_per_class=data.get("detectors_per_class", 20),
            max_generation_attempts=data.get("max_generation_attempts", 1000),
        )

        agent.global_threshold = data.get("global_threshold", 0.5)
        agent.training_stats = data.get("training_stats", {})

        agent.self_patterns_by_class = {
            label: [Pattern(**pattern) for pattern in patterns]
            for label, patterns in data.get("self_patterns_by_class", {}).items()
        }
        agent.self_embeddings_by_class = {
            label: [np.array(embedding) for embedding in embeddings]
            for label, embeddings in data.get("self_embeddings_by_class", {}).items()
        }

        agent.detector_sets = {}
        for label, detector_set in data.get("detector_sets", {}).items():
            set_obj = ClassDetectorSet(class_label=label, num_detectors=len(detector_set.get("detectors", [])))
            set_obj.threshold = detector_set.get("threshold", agent.global_threshold)
            set_obj.detectors = [np.array(detector) for detector in detector_set.get("detectors", [])]
            set_obj.quality_scores = detector_set.get("quality_scores", [])
            agent.detector_sets[label] = set_obj

        return agent
