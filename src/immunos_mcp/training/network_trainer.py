"""
Train IMMUNOS Enhanced NK agent on NSL-KDD (KDDTrain+) data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..agents.nk_cell_enhanced import EnhancedNKCellAgent
from ..config.paths import get_data_root, resolve_data_path
from ..core.antigen import Antigen


CATEGORICAL_INDEXES = {1, 2, 3}


@dataclass
class NetworkTrainingResult:
    normal_samples: int
    detectors: int
    feature_dim: int


def _kdd_default_path() -> Path:
    candidate = get_data_root() / "network" / "KDDTrain+.txt"
    return candidate


def _encode_categories(value: str, mapping: Dict[str, int]) -> int:
    if value not in mapping:
        mapping[value] = len(mapping) + 1
    return mapping[value]


def load_kdd_normal(path: Path, max_samples: int = 5000) -> Tuple[List[List[float]], List[Antigen]]:
    category_maps: Dict[int, Dict[str, int]] = {idx: {} for idx in CATEGORICAL_INDEXES}
    feature_rows: List[List[float]] = []
    antigens: List[Antigen] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) < 42:
                continue
            label = parts[-2].strip()
            if label != "normal":
                continue

            feature_values: List[float] = []
            for i, value in enumerate(parts[:-2]):
                if i in CATEGORICAL_INDEXES:
                    encoded = _encode_categories(value, category_maps[i])
                    feature_values.append(float(encoded))
                else:
                    try:
                        feature_values.append(float(value))
                    except ValueError:
                        feature_values.append(0.0)

            feature_rows.append(feature_values)
            antigens.append(Antigen.from_dict({"features": feature_values}, class_label="normal"))

            if max_samples and len(feature_rows) >= max_samples:
                break

    return feature_rows, antigens


def normalize_features(feature_rows: List[List[float]]) -> np.ndarray:
    data = np.array(feature_rows, dtype=np.float32)
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    normalized = (data - mins) / ranges
    return normalized


def train_network_agent(path: Path, max_samples: int) -> Tuple[EnhancedNKCellAgent, NetworkTrainingResult]:
    feature_rows, antigens = load_kdd_normal(path, max_samples=max_samples)
    if not feature_rows:
        raise ValueError("No normal samples found in dataset")

    normalized = normalize_features(feature_rows)
    embeddings = [row for row in normalized]

    nk_cell = EnhancedNKCellAgent(agent_name="network_nk_enhanced")
    nk_cell.train_on_self(antigens, embeddings=embeddings)

    detector_count = sum(len(ds.detectors) for ds in nk_cell.detector_sets.values())
    result = NetworkTrainingResult(
        normal_samples=len(antigens),
        detectors=detector_count,
        feature_dim=normalized.shape[1],
    )
    return nk_cell, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IMMUNOS Enhanced NK on NSL-KDD")
    parser.add_argument("--kdd", type=str, default=None,
                        help="Path to KDDTrain+.txt (absolute or relative to data root)")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max normal samples to load (0 = all)")
    parser.add_argument("--save-nk", type=str, default=None,
                        help="Optional path to save Enhanced NK state")
    args = parser.parse_args()

    data_path = resolve_data_path(args.kdd, None) if args.kdd else _kdd_default_path()
    if not data_path.exists():
        raise FileNotFoundError(f"KDD dataset not found at {data_path}")

    nk_cell, result = train_network_agent(data_path, max_samples=args.max_samples)
    if args.save_nk:
        nk_cell.save_state(args.save_nk)
    print("✅ Network training complete")
    print(f"  Normal samples: {result.normal_samples}")
    print(f"  Detectors generated: {result.detectors}")
    print(f"  Feature dimension: {result.feature_dim}")


if __name__ == "__main__":
    main()
