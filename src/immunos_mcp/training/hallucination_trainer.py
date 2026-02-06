"""
Train IMMUNOS agents for hallucination detection using TruthfulQA.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from ..agents.bcell_agent import BCellAgent
from ..agents.nk_cell_agent import NKCellAgent
from ..config.paths import get_data_root, resolve_data_path
from ..core.antigen import Antigen
from ..algorithms.negsel import NegativeSelectionClassifier, NEGSEL_PRESETS
from ..core.fusion import LLMModalityFusion
from ..embeddings.simple_text_embedder import SimpleTextEmbedder


@dataclass
class HallucinationTrainingResult:
    truthful_count: int
    hallucinated_count: int
    bcell_patterns: int
    nk_self_patterns: int


def _truthfulqa_default_path() -> Path:
    candidates = [
        get_data_root() / "hallucination" / "truthfulqa" / "TruthfulQA.csv",
        get_data_root() / "hallucination" / "truthfulqa" / "data" / "v1" / "TruthfulQA.csv",
        get_data_root() / "hallucination" / "truthfulqa" / "data" / "v0" / "TruthfulQA.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_truthfulqa(path: Path,
                    include_question: bool = True,
                    max_samples: int = 0) -> Tuple[List[Antigen], List[Antigen]]:
    truthful: List[Antigen] = []
    hallucinated: List[Antigen] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = (row.get("Question") or "").strip()
            best_answer = (row.get("Best Answer") or "").strip()
            incorrect = (row.get("Best Incorrect Answer") or "").strip()

            if not best_answer or not incorrect:
                continue

            if include_question and question:
                truthful_text = f"Q: {question}\nA: {best_answer}"
                hallucinated_text = f"Q: {question}\nA: {incorrect}"
            else:
                truthful_text = best_answer
                hallucinated_text = incorrect

            truthful.append(Antigen.from_text(truthful_text, class_label="truthful"))
            hallucinated.append(Antigen.from_text(hallucinated_text, class_label="hallucinated"))

            if max_samples and len(truthful) >= max_samples:
                break

    return truthful, hallucinated


def train_hallucination_agents(truthful: List[Antigen],
                               hallucinated: List[Antigen],
                               embedder: SimpleTextEmbedder,
                               nk_threshold: float,
                               nk_detectors: int) -> Tuple[BCellAgent, NKCellAgent, HallucinationTrainingResult]:
    bcell = BCellAgent(agent_name="hallucination_bcell")
    nk_cell = NKCellAgent(
        agent_name="hallucination_nk",
        detection_threshold=nk_threshold,
        num_detectors=nk_detectors,
    )

    all_antigens = truthful + hallucinated
    embeddings_all = [embedder.embed(antigen.get_text_content()) for antigen in all_antigens]
    embeddings_truthful = embeddings_all[:len(truthful)]
    bcell.train(all_antigens, embeddings=embeddings_all)
    nk_cell.train_on_self(truthful, embeddings=embeddings_truthful)

    result = HallucinationTrainingResult(
        truthful_count=len(truthful),
        hallucinated_count=len(hallucinated),
        bcell_patterns=len(bcell.patterns),
        nk_self_patterns=len(nk_cell.self_patterns),
    )
    return bcell, nk_cell, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IMMUNOS hallucination agents (TruthfulQA)")
    parser.add_argument("--truthfulqa", type=str, default=None,
                        help="Path to TruthfulQA CSV (absolute or relative to data root)")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max TruthfulQA samples to load (0 = all)")
    parser.add_argument("--no-question", action="store_true",
                        help="Use answers only (skip question context)")
    parser.add_argument("--save-bcell", type=str, default=None,
                        help="Optional path to save B Cell state")
    parser.add_argument("--save-nk", type=str, default=None,
                        help="Optional path to save NK state")
    parser.add_argument("--nk-threshold", type=float, default=0.8,
                        help="NK detection threshold for detector generation")
    parser.add_argument("--nk-detectors", type=int, default=100,
                        help="Number of NK detectors to generate")
    args = parser.parse_args()

    data_path = resolve_data_path(args.truthfulqa, None) if args.truthfulqa else _truthfulqa_default_path()
    if not data_path.exists():
        raise FileNotFoundError(f"TruthfulQA not found at {data_path}")

    truthful, hallucinated = load_truthfulqa(
        path=data_path,
        include_question=not args.no_question,
        max_samples=args.max_samples,
    )

    embedder = SimpleTextEmbedder()
    bcell, nk_cell, result = train_hallucination_agents(
        truthful,
        hallucinated,
        embedder,
        nk_threshold=args.nk_threshold,
        nk_detectors=args.nk_detectors,
    )
    if args.save_bcell:
        bcell.save_state(args.save_bcell)
    if args.save_nk:
        nk_cell.save_state(args.save_nk)
    print("✅ Hallucination training complete")
    print(f"  Truthful samples: {result.truthful_count}")
    print(f"  Hallucinated samples: {result.hallucinated_count}")
    print(f"  B Cell patterns: {result.bcell_patterns}")
    print(f"  NK self patterns: {result.nk_self_patterns}")


if __name__ == "__main__":
    main()
