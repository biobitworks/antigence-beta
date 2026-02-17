"""
B Cell Agent for IMMUNOS-MCP

Pattern matching agent inspired by B cells and antibody production.
Learns specific patterns and calculates affinity to new antigens.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..core.affinity import AffinityCalculator
from ..core.antigen import Antigen
from ..core.protocols import AgentResponse, Clone, Pattern, RecognitionResult


class BCellAgent:
    """
    B Cell Agent - Pattern Matcher

    Mimics B cell behavior:
    - Learns patterns from training examples (antibody production)
    - Calculates affinity to new antigens (binding strength)
    - Groups into clones for collective recognition (plasma cells)
    - Supports affinity maturation (improving pattern matching)
    """

    def __init__(
        self,
        agent_name: str = "bcell_001",
        affinity_method: str = "hybrid",
        embedding_weight: float = 0.7,
    ):
        """
        Initialize B Cell agent.

        Args:
            agent_name: Unique identifier for this agent
            affinity_method: "traditional", "embedding", or "hybrid"
            embedding_weight: Embedding weight (0-1) when using hybrid affinity
        """
        self.agent_name = agent_name
        self.affinity_calculator = AffinityCalculator(
            method=affinity_method,
            embedding_weight=embedding_weight,
        )
        self.patterns: List[Pattern] = []
        self.clones: Dict[str, Clone] = {}  # class_label -> Clone
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def train(self, training_data: List[Antigen], embeddings: Optional[List[np.ndarray]] = None):
        """
        Train on antigens (create patterns/antibodies).

        Args:
            training_data: List of training antigens with labels
            embeddings: Optional pre-computed embeddings
        """
        print(f"[{self.agent_name}] Training on {len(training_data)} antigens...")

        for i, antigen in enumerate(training_data):
            if antigen.class_label is None:
                continue

            # Create pattern
            pattern = Pattern(
                pattern_id=f"{self.agent_name}_pattern_{i}",
                class_label=antigen.class_label,
                example_data=antigen.data,
                embedding=embeddings[i].tolist() if embeddings and i < len(embeddings) else None,
                features=antigen.features,
                creation_time=time.time(),
            )

            self.patterns.append(pattern)

            # Cache embedding if provided
            if embeddings and i < len(embeddings):
                self.embedding_cache[pattern.pattern_id] = embeddings[i]

        # Group patterns into clones by class
        self._create_clones()

        print(
            f"[{self.agent_name}] Created {len(self.patterns)} patterns in {len(self.clones)} clones"
        )

    def _create_clones(self):
        """Group patterns into clones by class label"""
        from collections import defaultdict

        patterns_by_class = defaultdict(list)
        for pattern in self.patterns:
            patterns_by_class[pattern.class_label].append(pattern)

        for class_label, class_patterns in patterns_by_class.items():
            clone = Clone(
                clone_id=f"{self.agent_name}_clone_{class_label}",
                class_label=class_label,
                patterns=class_patterns,
                size=len(class_patterns),
            )
            self.clones[class_label] = clone

    def calculate_affinity(
        self, antigen: Antigen, pattern: Pattern, antigen_embedding: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate affinity between antigen and pattern.

        Args:
            antigen: Test antigen
            pattern: Learned pattern
            antigen_embedding: Optional embedding for antigen

        Returns:
            Affinity score (0-1)
        """
        pattern_embedding = None
        if pattern.pattern_id in self.embedding_cache:
            pattern_embedding = self.embedding_cache[pattern.pattern_id]

        result = self.affinity_calculator.calculate(
            antigen.data,
            pattern.example_data,
            embeddings1=antigen_embedding,
            embeddings2=pattern_embedding,
        )

        return result.score

    def recognize(
        self,
        antigen: Antigen,
        antigen_embedding: Optional[np.ndarray] = None,
        strategy: str = "sha",
    ) -> RecognitionResult:
        """
        Recognize antigen using learned patterns.
        Parallelized using PyTorch if embeddings are available.
        """
        start_time = time.time()
        avidity_scores = {}

        if antigen_embedding is not None and self.affinity_calculator.method in [
            "embedding",
            "hybrid",
        ]:
            # 1. Prepare antigen tensor
            antigen_tensor = torch.from_numpy(antigen_embedding).float().unsqueeze(0)  # (1, Dim)

            for class_label, clone in self.clones.items():
                # 2. Extract pattern embeddings for this clone
                clone_embeddings = [p.embedding for p in clone.patterns if p.embedding is not None]
                if not clone_embeddings:
                    avidity_scores[class_label] = 0.0
                    continue

                pattern_tensor = torch.tensor(clone_embeddings).float()  # (M, Dim)

                # 3. Batch affinity calculation
                affinities = self.affinity_calculator.torch_batch_affinity(
                    antigen_tensor, pattern_tensor
                )

                # 4. Calculate avidity (collective affinity)
                # affinities shape is (1, M), take first row
                avidity = clone.calculate_avidity(affinities[0].tolist())
                avidity_scores[class_label] = avidity
        else:
            # Fallback to sequential traditional calculation
            for class_label, clone in self.clones.items():
                affinities = [
                    self.calculate_affinity(antigen, p, antigen_embedding) for p in clone.patterns
                ]
                avidity = clone.calculate_avidity(affinities)
                avidity_scores[class_label] = avidity

        # Apply recognition strategy
        if strategy == "sha":
            result = self._apply_sha(avidity_scores)
        elif strategy == "rha":
            result = self._apply_rha(avidity_scores)
        else:
            result = self._apply_sha(avidity_scores)

        result.agents_involved = [self.agent_name]
        result.metadata["execution_time"] = time.time() - start_time
        result.metadata["num_patterns"] = len(self.patterns)
        result.metadata["num_clones"] = len(self.clones)

        return result

    def _apply_sha(self, avidity_scores: Dict[str, float]) -> RecognitionResult:
        """Simple Highest Avidity strategy"""
        if not avidity_scores:
            return RecognitionResult(
                predicted_class=None,
                confidence=0.0,
                is_uncertain=True,
                avidity_scores=avidity_scores,
                explanation="No patterns available",
            )

        winner_class = max(avidity_scores, key=avidity_scores.get)
        winner_avidity = avidity_scores[winner_class]
        total_avidity = sum(avidity_scores.values())

        confidence = winner_avidity / total_avidity if total_avidity > 0 else 0.0

        return RecognitionResult(
            predicted_class=winner_class,
            confidence=confidence,
            is_uncertain=False,
            avidity_scores=avidity_scores,
            explanation=f"Highest avidity: {winner_class} ({winner_avidity:.3f})",
        )

    def _apply_rha(
        self, avidity_scores: Dict[str, float], threshold: float = 0.05
    ) -> RecognitionResult:
        """Relative Highest Avidity strategy with 5% threshold"""
        if not avidity_scores:
            return RecognitionResult(
                predicted_class=None,
                confidence=0.0,
                is_uncertain=True,
                avidity_scores=avidity_scores,
            )

        sorted_classes = sorted(avidity_scores.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_classes) < 2:
            first_class, first_avidity = sorted_classes[0]
            return RecognitionResult(
                predicted_class=first_class,
                confidence=1.0,
                is_uncertain=False,
                avidity_scores=avidity_scores,
                explanation=f"Only one class available: {first_class}",
            )

        first_class, first_avidity = sorted_classes[0]
        second_class, second_avidity = sorted_classes[1]

        # Calculate relative difference
        total = first_avidity + second_avidity
        relative_diff = (first_avidity - second_avidity) / total if total > 0 else 0

        if relative_diff >= threshold:
            # Clear winner
            return RecognitionResult(
                predicted_class=first_class,
                confidence=relative_diff,
                is_uncertain=False,
                avidity_scores=avidity_scores,
                explanation=f"Clear winner: {first_class} vs {second_class} ({relative_diff:.3f})",
            )
        else:
            # Too close to call
            return RecognitionResult(
                predicted_class=first_class,
                confidence=relative_diff,
                is_uncertain=True,
                avidity_scores=avidity_scores,
                explanation=f"Uncertain: {first_class} vs {second_class} ({relative_diff:.3f} < {threshold})",
            )

    def add_pattern(self, antigen: Antigen, embedding: Optional[np.ndarray] = None):
        """
        Add new pattern online (incremental learning).

        Args:
            antigen: New training example
            embedding: Optional embedding
        """
        if antigen.class_label is None:
            raise ValueError("Antigen must have class label for training")

        pattern = Pattern(
            pattern_id=f"{self.agent_name}_pattern_{len(self.patterns)}",
            class_label=antigen.class_label,
            example_data=antigen.data,
            embedding=embedding.tolist() if embedding is not None else None,
            features=antigen.features,
            creation_time=time.time(),
        )

        self.patterns.append(pattern)

        if embedding is not None:
            self.embedding_cache[pattern.pattern_id] = embedding

        # Update or create clone
        if antigen.class_label in self.clones:
            self.clones[antigen.class_label].patterns.append(pattern)
            self.clones[antigen.class_label].size += 1
        else:
            self.clones[antigen.class_label] = Clone(
                clone_id=f"{self.agent_name}_clone_{antigen.class_label}",
                class_label=antigen.class_label,
                patterns=[pattern],
                size=1,
            )

    def mature_affinity(
        self, antigen: Antigen, embedding: Optional[np.ndarray] = None, iterations: int = 3
    ) -> Pattern:
        """
        Affinity maturation: improve pattern matching through mutation and selection.

        Args:
            antigen: Target antigen
            embedding: Optional embedding
            iterations: Number of maturation iterations

        Returns:
            Best matured pattern
        """
        # Find best matching pattern
        best_pattern = None
        best_affinity = 0.0

        for pattern in self.patterns:
            affinity = self.calculate_affinity(antigen, pattern, embedding)
            if affinity > best_affinity:
                best_affinity = affinity
                best_pattern = pattern

        # Return original if no improvement possible
        return best_pattern if best_pattern else self.patterns[0] if self.patterns else None

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "agent_name": self.agent_name,
            "num_patterns": len(self.patterns),
            "num_clones": len(self.clones),
            "classes": list(self.clones.keys()),
            "patterns_per_class": {cls: clone.size for cls, clone in self.clones.items()},
        }

    def to_agent_response(
        self, result: RecognitionResult, success: bool = True, execution_time: float = 0.0
    ) -> AgentResponse:
        """Convert recognition result to agent response"""
        return AgentResponse(
            agent_name=self.agent_name,
            agent_type="bcell",
            result=result.to_dict(),
            success=success,
            execution_time=execution_time,
            metadata=result.metadata,
        )

    def save_state(self, path: str) -> None:
        """Persist patterns and embeddings to disk."""
        state = {
            "agent_name": self.agent_name,
            "affinity_method": self.affinity_calculator.method,
            "patterns": [pattern.to_dict() for pattern in self.patterns],
            "embedding_cache": {key: value.tolist() for key, value in self.embedding_cache.items()},
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load_state(cls, path: str) -> "BCellAgent":
        """Load patterns and embeddings from disk."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            agent_name=data.get("agent_name", "bcell_001"),
            affinity_method=data.get("affinity_method", "hybrid"),
        )
        agent.patterns = [Pattern(**pattern) for pattern in data.get("patterns", [])]
        agent.embedding_cache = {
            key: np.array(value) for key, value in data.get("embedding_cache", {}).items()
        }
        agent._create_clones()
        return agent
