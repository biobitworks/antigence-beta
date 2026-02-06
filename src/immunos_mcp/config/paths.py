"""
Shared path helpers for local datasets and persistence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_DATA_ROOT = Path("./data/immunos_data")


def get_data_root(override: Optional[str] = None) -> Path:
    """
    Resolve the dataset root directory.

    Order: explicit override -> IMMUNOS_DATA_ROOT -> DEFAULT_DATA_ROOT.
    """
    value = override or os.getenv("IMMUNOS_DATA_ROOT")
    return Path(value) if value else DEFAULT_DATA_ROOT


def resolve_data_path(path: str, data_root_override: Optional[str] = None) -> Path:
    """
    Resolve a dataset path that may be relative to the data root.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return get_data_root(data_root_override) / candidate
