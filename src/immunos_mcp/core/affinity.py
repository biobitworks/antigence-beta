"""
Affinity calculation for IMMUNOS-MCP

Combines traditional Immunos-81 affinity calculations with
LLM embedding-based semantic similarity.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class AffinityResult:
    """Result of affinity calculation"""

    score: float
    method: str
    components: Dict[str, float] = None
    confidence: float = 1.0

    def __float__(self) -> float:
        return self.score


class AffinityCalculator:
    """
    Calculate affinity (similarity) between antigens and patterns.

    Supports multiple calculation methods:
    - Traditional: Exponential decay for numeric, exact match for nominal
    - Embedding: Cosine similarity of LLM embeddings
    - Hybrid: Weighted combination of traditional + embedding
    """

    def __init__(self, method: str = "hybrid", embedding_weight: float = 0.7):
        """
        Initialize affinity calculator.

        Args:
            method: "traditional", "embedding", or "hybrid"
            embedding_weight: Weight for embedding similarity (0-1) in hybrid mode
        """
        self.method = method
        self.embedding_weight = embedding_weight
        self.traditional_weight = 1.0 - embedding_weight

    def calculate(
        self,
        antigen1: Any,
        antigen2: Any,
        embeddings1: Optional[np.ndarray] = None,
        embeddings2: Optional[np.ndarray] = None,
    ) -> AffinityResult:
        """
        Calculate affinity between two antigens.

        Args:
            antigen1: First antigen
            antigen2: Second antigen
            embeddings1: Optional embedding vector for antigen1
            embeddings2: Optional embedding vector for antigen2

        Returns:
            AffinityResult with score and metadata
        """
        if self.method == "traditional":
            return self._traditional_affinity(antigen1, antigen2)
        elif self.method == "embedding":
            if embeddings1 is None or embeddings2 is None:
                raise ValueError("Embeddings required for embedding method")
            return self._embedding_affinity(embeddings1, embeddings2)
        elif self.method == "hybrid":
            traditional = self._traditional_affinity(antigen1, antigen2)
            if embeddings1 is not None and embeddings2 is not None:
                embedding = self._embedding_affinity(embeddings1, embeddings2)
                score = (
                    self.traditional_weight * traditional.score
                    + self.embedding_weight * embedding.score
                )
                return AffinityResult(
                    score=score,
                    method="hybrid",
                    components={"traditional": traditional.score, "embedding": embedding.score},
                )
            else:
                return traditional
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _traditional_affinity(self, val1: Any, val2: Any) -> AffinityResult:
        """
        Traditional Immunos-81 affinity calculation.

        For numeric: exp(-|v1 - v2| / sigma)
        For nominal: 1.0 if equal, 0.0 otherwise
        For text: Character-level distance
        """
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Numeric affinity with exponential decay
            distance = abs(val1 - val2)
            sigma = max(abs(val1), abs(val2), 1.0) * 0.1  # 10% of magnitude
            affinity = math.exp(-distance / sigma)
            return AffinityResult(score=affinity, method="traditional_numeric")

        elif isinstance(val1, str) and isinstance(val2, str):
            # String affinity (simplified)
            if val1 == val2:
                return AffinityResult(score=1.0, method="traditional_exact")
            else:
                # Simple character overlap
                set1, set2 = set(val1.lower()), set(val2.lower())
                if not set1 or not set2:
                    return AffinityResult(score=0.0, method="traditional_string")
                overlap = len(set1 & set2)
                union = len(set1 | set2)
                affinity = overlap / union if union > 0 else 0.0
                return AffinityResult(score=affinity, method="traditional_string")

        elif isinstance(val1, dict) and isinstance(val2, dict):
            # Structured data affinity
            common_keys = set(val1.keys()) & set(val2.keys())
            if not common_keys:
                return AffinityResult(score=0.0, method="traditional_dict")

            affinities = []
            for key in common_keys:
                sub_result = self._traditional_affinity(val1[key], val2[key])
                affinities.append(sub_result.score)

            avg_affinity = sum(affinities) / len(affinities) if affinities else 0.0
            return AffinityResult(score=avg_affinity, method="traditional_dict")

        else:
            # Default: exact match
            return AffinityResult(score=1.0 if val1 == val2 else 0.0, method="traditional_default")

    def _embedding_affinity(self, emb1: np.ndarray, emb2: np.ndarray) -> AffinityResult:
        """
        Embedding-based affinity using cosine similarity.

        Args:
            emb1: Embedding vector 1
            emb2: Embedding vector 2

        Returns:
            AffinityResult with cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return AffinityResult(score=0.0, method="embedding_cosine")

        # Cosine similarity
        cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)

        # Convert from [-1, 1] to [0, 1]
        affinity = (cosine_sim + 1.0) / 2.0

        return AffinityResult(score=float(affinity), method="embedding_cosine")

    def torch_batch_affinity(self, emb1: "torch.Tensor", emb2: "torch.Tensor") -> "torch.Tensor":
        """
        Calculate cosine similarity between two batches of embeddings using PyTorch.
        Returns a matrix of scores in [0, 1].

        Args:
            emb1: Tensor of shape (N, Dim)
            emb2: Tensor of shape (M, Dim)
        """
        # Normalize
        emb1_norm = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2_norm = torch.nn.functional.normalize(emb2, p=2, dim=1)

        # Cosine similarity: (N, Dim) @ (Dim, M) -> (N, M)
        cosine_sim = torch.mm(emb1_norm, emb2_norm.t())

        # Scale to [0, 1]
        return (cosine_sim + 1.0) / 2.0

    def calculate_avidity(self, affinities: List[float], clone_size: int) -> float:
        """
        Calculate avidity (collective affinity) for a clone.

        Uses Immunos-81 formula: sum(affinities) * log(1 + clone_size)

        Args:
            affinities: List of affinity scores
            clone_size: Number of members in clone

        Returns:
            Avidity score
        """
        if not affinities:
            return 0.0

        total_affinity = sum(affinities)
        concentration_factor = math.log(1 + clone_size)

        return total_affinity * concentration_factor


class DistanceMetric:
    """Distance metrics for pattern matching"""

    @staticmethod
    def euclidean(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Euclidean distance"""
        return float(np.linalg.norm(vec1 - vec2))

    @staticmethod
    def manhattan(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Manhattan (L1) distance"""
        return float(np.sum(np.abs(vec1 - vec2)))

    @staticmethod
    def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine distance (1 - cosine similarity)"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(1.0 - (cosine_sim + 1.0) / 2.0)

    @staticmethod
    def hamming(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Hamming distance (for binary vectors)"""
        return float(np.sum(vec1 != vec2))
