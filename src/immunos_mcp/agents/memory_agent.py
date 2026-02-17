"""
Memory Agent for IMMUNOS

Implements T Cell-like memory with:
- Adaptive decay (priority-based retention)
- Few-shot retrieval for similar past cases
- Memory consolidation (pruning low-value memories)
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    priority: str = "medium"  # low, medium, high, critical
    embedding: Optional[List[float]] = None
    class_label: Optional[str] = None
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryEntry":
        return cls(**data)

    def relevance_score(self, current_time: float, decay_rate: float = 0.1) -> float:
        """
        Calculate relevance score with adaptive decay.

        Higher priority = slower decay
        More accesses = higher base score
        """
        priority_multiplier = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }.get(self.priority, 0.5)

        age_days = (current_time - self.last_accessed) / 86400
        decay = np.exp(-decay_rate * age_days / priority_multiplier)

        # Base score from access count and confidence
        base_score = min(1.0, (self.access_count / 10) + self.confidence)

        return base_score * decay


class MemoryAgent:
    """
    T Cell-like memory with adaptive decay and few-shot retrieval.

    Features:
    - Priority-based retention (critical > high > medium > low)
    - Cosine similarity for few-shot retrieval
    - Automatic consolidation (prune low-relevance memories)
    - Persistent storage with JSON
    """

    def __init__(
        self,
        agent_name: str = "memory_001",
        path: Optional[Path] = None,
        max_memories: int = 10000,
        decay_rate: float = 0.1,
        consolidation_threshold: float = 0.1
    ):
        self.agent_name = agent_name
        self.path = path or Path(".immunos") / "memory" / "immunos_memory.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_memories = max_memories
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold
        self._store: Dict[str, MemoryEntry] = self._load()

    def store(
        self,
        key: str,
        value: Any,
        priority: str = "medium",
        embedding: Optional[List[float]] = None,
        class_label: Optional[str] = None,
        confidence: float = 0.0,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Store a memory with metadata.

        Args:
            key: Unique identifier
            value: Data to store
            priority: Retention priority (low/medium/high/critical)
            embedding: Optional vector for similarity search
            class_label: Classification result if applicable
            confidence: Confidence score (0-1)
            tags: Searchable tags
        """
        entry = MemoryEntry(
            key=key,
            value=value,
            priority=priority,
            embedding=embedding,
            class_label=class_label,
            confidence=confidence,
            tags=tags or []
        )
        self._store[key] = entry

        # Consolidate if over limit
        if len(self._store) > self.max_memories:
            self.consolidate()

        self._save()

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a stored value by key, updating access metadata."""
        if key not in self._store:
            return None

        entry = self._store[key]
        entry.last_accessed = time.time()
        entry.access_count += 1
        self._save()

        return entry.value

    def retrieve_similar(
        self,
        query_embedding: List[float],
        k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Few-shot retrieval: find k most similar memories.

        Args:
            query_embedding: Vector to match against
            k: Number of results to return
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of (MemoryEntry, similarity_score) tuples
        """
        if not query_embedding:
            return []

        query = np.array(query_embedding)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        results: List[Tuple[MemoryEntry, float]] = []

        for entry in self._store.values():
            if entry.embedding is None:
                continue

            mem = np.array(entry.embedding)
            mem_norm = np.linalg.norm(mem)
            if mem_norm == 0:
                continue

            similarity = np.dot(query, mem) / (query_norm * mem_norm)

            if similarity >= min_similarity:
                # Update access metadata
                entry.last_accessed = time.time()
                entry.access_count += 1
                results.append((entry, float(similarity)))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        if results:
            self._save()

        return results[:k]

    def retrieve_by_class(
        self,
        class_label: str,
        k: int = 10
    ) -> List[MemoryEntry]:
        """Retrieve memories by class label."""
        matches = [
            entry for entry in self._store.values()
            if entry.class_label == class_label
        ]
        # Sort by relevance
        current_time = time.time()
        matches.sort(
            key=lambda e: e.relevance_score(current_time, self.decay_rate),
            reverse=True
        )
        return matches[:k]

    def retrieve_by_tags(
        self,
        tags: List[str],
        match_all: bool = False
    ) -> List[MemoryEntry]:
        """Retrieve memories by tags."""
        tag_set = set(tags)
        matches = []

        for entry in self._store.values():
            entry_tags = set(entry.tags)
            if match_all:
                if tag_set <= entry_tags:
                    matches.append(entry)
            else:
                if tag_set & entry_tags:
                    matches.append(entry)

        return matches

    def decay(self) -> int:
        """
        Apply adaptive decay, removing low-relevance memories.

        Returns number of memories removed.
        """
        current_time = time.time()
        to_remove = []

        for key, entry in self._store.items():
            score = entry.relevance_score(current_time, self.decay_rate)
            if score < self.consolidation_threshold:
                to_remove.append(key)

        for key in to_remove:
            del self._store[key]

        if to_remove:
            self._save()

        return len(to_remove)

    def consolidate(self) -> int:
        """
        Consolidate memories by removing lowest-relevance entries.

        Keeps max_memories entries, prioritizing high-relevance.
        Returns number of memories removed.
        """
        if len(self._store) <= self.max_memories:
            return 0

        current_time = time.time()

        # Score all memories
        scored = [
            (key, entry.relevance_score(current_time, self.decay_rate))
            for key, entry in self._store.items()
        ]

        # Sort by score ascending (lowest first)
        scored.sort(key=lambda x: x[1])

        # Remove lowest until under limit
        to_remove = len(self._store) - self.max_memories
        removed = 0

        for key, _ in scored[:to_remove]:
            del self._store[key]
            removed += 1

        self._save()
        return removed

    def store_classification(
        self,
        antigen_hash: str,
        antigen_text: str,
        class_label: str,
        confidence: float,
        embedding: Optional[List[float]] = None,
        features: Optional[Dict] = None
    ) -> None:
        """
        Store a successful classification for future few-shot learning.

        Args:
            antigen_hash: Unique hash of the antigen
            antigen_text: Original text
            class_label: Classification result
            confidence: Confidence score
            embedding: Vector representation
            features: Dendritic features
        """
        priority = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"

        self.store(
            key=f"classification_{antigen_hash}",
            value={
                "text": antigen_text[:500],  # Truncate for storage
                "class_label": class_label,
                "confidence": confidence,
                "features": features
            },
            priority=priority,
            embedding=embedding,
            class_label=class_label,
            confidence=confidence,
            tags=["classification", class_label]
        )

    def get_few_shot_examples(
        self,
        query_embedding: List[float],
        k: int = 3
    ) -> List[Dict]:
        """
        Get few-shot examples for a query.

        Returns list of similar past classifications for context.
        """
        similar = self.retrieve_similar(query_embedding, k=k, min_similarity=0.6)

        examples = []
        for entry, similarity in similar:
            if isinstance(entry.value, dict) and "text" in entry.value:
                examples.append({
                    "text": entry.value["text"],
                    "class_label": entry.class_label,
                    "confidence": entry.confidence,
                    "similarity": similarity
                })

        return examples

    def clear(self) -> None:
        """Clear all stored memory."""
        self._store = {}
        self._save()

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        time.time()
        priorities = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        classes: Dict[str, int] = {}
        total_accesses = 0

        for entry in self._store.values():
            priorities[entry.priority] = priorities.get(entry.priority, 0) + 1
            if entry.class_label:
                classes[entry.class_label] = classes.get(entry.class_label, 0) + 1
            total_accesses += entry.access_count

        return {
            "agent_name": self.agent_name,
            "total_memories": len(self._store),
            "max_memories": self.max_memories,
            "by_priority": priorities,
            "by_class": classes,
            "total_accesses": total_accesses,
            "decay_rate": self.decay_rate,
            "consolidation_threshold": self.consolidation_threshold
        }

    def _load(self) -> Dict[str, MemoryEntry]:
        """Load memory from disk if available."""
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                return {
                    key: MemoryEntry.from_dict(entry)
                    for key, entry in data.items()
                }
        except (json.JSONDecodeError, OSError, TypeError):
            return {}

    def _save(self) -> None:
        """Persist memory to disk."""
        with self.path.open("w", encoding="utf-8") as handle:
            data = {key: entry.to_dict() for key, entry in self._store.items()}
            json.dump(data, handle, indent=2)


def generate_hash(text: str) -> str:
    """Generate stable hash for text content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
