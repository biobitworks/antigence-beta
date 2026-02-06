"""
Train IMMUNOS agents for emotion detection using numeric feature vectors.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..agents.bcell_agent import BCellAgent
from ..agents.nk_cell_agent import NKCellAgent
from ..config.paths import get_data_root, resolve_data_path
from ..core.antigen import Antigen


@dataclass
class EmotionTrainingResult:
    samples_used: int
    bcell_patterns: int
    nk_self_patterns: int


def _emotion_default_path() -> Path:
    return get_data_root() / "emotion" / "features"


def _load_csv_features(path: Path, max_samples: int) -> List[np.ndarray]:
    vectors: List[np.ndarray] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            values = []
            for item in row:
                try:
                    values.append(float(item))
                except ValueError:
                    continue
            if values:
                vectors.append(np.array(values, dtype=np.float32))
            if max_samples and len(vectors) >= max_samples:
                break
    return vectors


def _load_npy_features(path: Path, max_samples: int) -> List[np.ndarray]:
    data = np.load(path)
    if data.ndim == 1:
        vectors = [data.astype(np.float32)]
    else:
        vectors = [row.astype(np.float32) for row in data]
    if max_samples:
        vectors = vectors[:max_samples]
    return vectors


def load_feature_vectors(path: Path, max_samples: int = 0) -> List[np.ndarray]:
    if path.is_dir():
        vectors: List[np.ndarray] = []
        for npy in sorted(path.glob("*.npy")):
            vectors.extend(_load_npy_features(npy, max_samples=0))
            if max_samples and len(vectors) >= max_samples:
                return vectors[:max_samples]
        return vectors

    if path.suffix.lower() == ".npy":
        return _load_npy_features(path, max_samples=max_samples)
    if path.suffix.lower() in {".csv", ".txt"}:
        return _load_csv_features(path, max_samples=max_samples)

    raise ValueError("Unsupported emotion feature format. Use .npy or .csv")


def train_emotion_agents(vectors: List[np.ndarray],
                         nk_threshold: float,
                         nk_detectors: int) -> Tuple[BCellAgent, NKCellAgent, EmotionTrainingResult]:
    bcell = BCellAgent(agent_name="emotion_bcell")
    nk_cell = NKCellAgent(
        agent_name="emotion_nk",
        detection_threshold=nk_threshold,
        num_detectors=nk_detectors,
    )

    antigens = [Antigen.from_dict({"features": vec.tolist()}, class_label="self") for vec in vectors]
    embeddings = vectors
    bcell.train(antigens, embeddings=embeddings)
    nk_cell.train_on_self(antigens, embeddings=embeddings)

    result = EmotionTrainingResult(
        samples_used=len(antigens),
        bcell_patterns=len(bcell.patterns),
        nk_self_patterns=len(nk_cell.self_patterns),
    )
    return bcell, nk_cell, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IMMUNOS emotion agents from feature vectors")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to feature file (.npy/.csv) or directory")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples to load (0 = all)")
    parser.add_argument("--save-bcell", type=str, default=None,
                        help="Optional path to save B Cell state")
    parser.add_argument("--save-nk", type=str, default=None,
                        help="Optional path to save NK state")
    parser.add_argument("--nk-threshold", type=float, default=0.8,
                        help="NK detection threshold for detector generation")
    parser.add_argument("--nk-detectors", type=int, default=100,
                        help="Number of NK detectors to generate")
    args = parser.parse_args()

    data_path = resolve_data_path(args.features, None) if args.features else _emotion_default_path()
    if not data_path.exists():
        raise FileNotFoundError(f"Emotion feature dataset not found at {data_path}")

    vectors = load_feature_vectors(data_path, max_samples=args.max_samples)
    if not vectors:
        raise ValueError("No emotion feature vectors loaded")

    bcell, nk_cell, result = train_emotion_agents(
        vectors,
        nk_threshold=args.nk_threshold,
        nk_detectors=args.nk_detectors,
    )

    if args.save_bcell:
        bcell.save_state(args.save_bcell)
    if args.save_nk:
        nk_cell.save_state(args.save_nk)

    print("✅ Emotion training complete")
    print(f"  Samples used: {result.samples_used}")
    print(f"  B Cell patterns: {result.bcell_patterns}")
    print(f"  NK self patterns: {result.nk_self_patterns}")


if __name__ == "__main__":
    main()
