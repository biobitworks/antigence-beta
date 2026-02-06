"""
Antigen representation for IMMUNOS-MCP

An antigen is a data unit to be processed by the immune system,
representing text, code, structured data, or other inputs.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class DataType(Enum):
    """Type of data contained in the antigen"""
    TEXT = "text"
    CODE = "code"
    STRUCTURED = "structured"
    IMAGE = "image"
    MIXED = "mixed"


@dataclass
class Antigen:
    """
    Represents a data unit to be classified or analyzed.

    Attributes:
        data: The raw data (text, code, dict, etc.)
        data_type: Type of data
        features: Extracted feature vector
        metadata: Additional context (sender, timestamp, etc.)
        class_label: Known classification (for training)
        identifier: Unique ID
    """
    data: Any
    data_type: DataType
    features: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    class_label: Optional[str] = None
    identifier: Optional[str] = None

    @classmethod
    def from_text(cls, text: str, class_label: Optional[str] = None,
                  metadata: Optional[Dict] = None) -> "Antigen":
        """
        Create antigen from text.

        Args:
            text: Text content
            class_label: Optional known classification
            metadata: Additional context

        Returns:
            Antigen instance
        """
        return cls(
            data=text,
            data_type=DataType.TEXT,
            metadata=metadata or {},
            class_label=class_label
        )

    @classmethod
    def from_code(cls, code: str, language: str = "python",
                  class_label: Optional[str] = None,
                  metadata: Optional[Dict] = None) -> "Antigen":
        """
        Create antigen from code.

        Args:
            code: Source code
            language: Programming language
            class_label: Optional known classification
            metadata: Additional context

        Returns:
            Antigen instance
        """
        meta = metadata or {}
        meta["language"] = language

        return cls(
            data=code,
            data_type=DataType.CODE,
            metadata=meta,
            class_label=class_label
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], class_label: Optional[str] = None,
                  metadata: Optional[Dict] = None) -> "Antigen":
        """
        Create antigen from structured data.

        Args:
            data: Structured data dictionary
            class_label: Optional known classification
            metadata: Additional context

        Returns:
            Antigen instance
        """
        return cls(
            data=data,
            data_type=DataType.STRUCTURED,
            metadata=metadata or {},
            class_label=class_label
        )

    def set_features(self, features: Dict[str, Any]):
        """Set extracted features"""
        self.features = features

    def add_metadata(self, key: str, value: Any):
        """Add metadata field"""
        self.metadata[key] = value

    def get_text_content(self) -> str:
        """Get text representation of data"""
        if self.data_type == DataType.TEXT:
            return str(self.data)
        elif self.data_type == DataType.CODE:
            return str(self.data)
        elif self.data_type == DataType.STRUCTURED:
            return str(self.data)
        else:
            return str(self.data)

    def __repr__(self) -> str:
        return f"Antigen(type={self.data_type.value}, label={self.class_label}, id={self.identifier})"


@dataclass
class AntigenBatch:
    """Batch of antigens for efficient processing"""
    antigens: List[Antigen]
    batch_metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.antigens)

    def __iter__(self):
        return iter(self.antigens)

    def __getitem__(self, idx: int) -> Antigen:
        return self.antigens[idx]

    @classmethod
    def from_texts(cls, texts: List[str], labels: Optional[List[str]] = None) -> "AntigenBatch":
        """Create batch from list of texts"""
        antigens = []
        for i, text in enumerate(texts):
            label = labels[i] if labels else None
            antigens.append(Antigen.from_text(text, class_label=label, metadata={"batch_idx": i}))
        return cls(antigens=antigens)
