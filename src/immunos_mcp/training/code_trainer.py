"""
Train IMMUNOS agents for code safety using local source files or DiverseVul dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from ..agents.bcell_agent import BCellAgent
from ..agents.nk_cell_agent import NKCellAgent
from ..config.paths import get_data_root, resolve_data_path
from ..core.antigen import Antigen
from ..embeddings.simple_text_embedder import SimpleTextEmbedder


CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".ps1"
}


@dataclass
class CodeTrainingResult:
    files_used: int
    bcell_patterns: int
    nk_self_patterns: int


def _code_default_path() -> Path:
    return get_data_root() / "code"


def load_code_samples(path: Path, max_samples: int = 0) -> List[Antigen]:
    antigens: List[Antigen] = []
    if path.is_file():
        paths = [path]
    else:
        paths = [p for p in path.rglob("*") if p.suffix in CODE_EXTENSIONS]

    for file_path in paths:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not content.strip():
            continue
        antigens.append(Antigen.from_text(content, class_label="safe", metadata={"path": str(file_path)}))
        if max_samples and len(antigens) >= max_samples:
            break
    return antigens


def load_diversevul(path: Path, max_samples: int = 0, balance: bool = True) -> Tuple[List[Antigen], List[Antigen]]:
    """
    Load DiverseVul dataset from JSONL file.

    Returns (safe_samples, vulnerable_samples) as Antigen lists.

    DiverseVul JSONL format (one JSON per line):
    {"func": "code", "target": 0 or 1 (1=vulnerable), "project": "name", ...}

    Note: DiverseVul file is sorted (vulnerable first, safe last), so we
    read samples from both ends to get balanced classes.
    """
    safe_samples: List[Antigen] = []
    vulnerable_samples: List[Antigen] = []

    json_files = list(path.glob("*.json")) if path.is_dir() else [path]
    target_per_class = (max_samples // 2) if max_samples else 10000

    for json_file in json_files:
        try:
            # Read file line by line to handle large files
            with open(json_file, encoding="utf-8") as f:
                lines = f.readlines()

            print(f"[code_trainer] Loading from {json_file.name} ({len(lines)} lines)")

            # Sample from beginning (vulnerable) and end (safe)
            vuln_lines = lines[:min(len(lines)//2, target_per_class * 2)]
            safe_lines = lines[-min(len(lines)//2, target_per_class * 2):]

            for line in vuln_lines:
                if len(vulnerable_samples) >= target_per_class:
                    break
                try:
                    item = json.loads(line.strip())
                    code = item.get("func", "")
                    if not code or len(code) < 20:
                        continue
                    if item.get("target", 0) == 1:
                        metadata = {"project": item.get("project", "unknown"), "source": "diversevul"}
                        vulnerable_samples.append(
                            Antigen.from_text(code, class_label="vulnerable", metadata=metadata)
                        )
                except json.JSONDecodeError:
                    continue

            for line in safe_lines:
                if len(safe_samples) >= target_per_class:
                    break
                try:
                    item = json.loads(line.strip())
                    code = item.get("func", "")
                    if not code or len(code) < 20:
                        continue
                    if item.get("target", 0) == 0:
                        metadata = {"project": item.get("project", "unknown"), "source": "diversevul"}
                        safe_samples.append(
                            Antigen.from_text(code, class_label="safe", metadata=metadata)
                        )
                except json.JSONDecodeError:
                    continue

        except OSError as e:
            print(f"[code_trainer] Warning: Could not read {json_file}: {e}")
            continue

    # Balance classes if requested
    if balance and safe_samples and vulnerable_samples:
        min_count = min(len(safe_samples), len(vulnerable_samples))
        safe_samples = safe_samples[:min_count]
        vulnerable_samples = vulnerable_samples[:min_count]

    return safe_samples, vulnerable_samples


def train_code_agents(antigens: List[Antigen],
                      embedder: SimpleTextEmbedder,
                      nk_threshold: float,
                      nk_detectors: int) -> Tuple[BCellAgent, NKCellAgent, CodeTrainingResult]:
    bcell = BCellAgent(agent_name="code_bcell")
    nk_cell = NKCellAgent(
        agent_name="code_nk",
        detection_threshold=nk_threshold,
        num_detectors=nk_detectors,
    )

    embeddings = [embedder.embed(antigen.get_text_content()) for antigen in antigens]
    bcell.train(antigens, embeddings=embeddings)
    nk_cell.train_on_self(antigens, embeddings=embeddings)

    result = CodeTrainingResult(
        files_used=len(antigens),
        bcell_patterns=len(bcell.patterns),
        nk_self_patterns=len(nk_cell.self_patterns),
    )
    return bcell, nk_cell, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IMMUNOS code agents on local source files or DiverseVul")
    parser.add_argument("--code-dir", type=str, default=None,
                        help="Path to code directory (absolute or relative to data root)")
    parser.add_argument("--diversevul-path", type=str, default=None,
                        help="Path to DiverseVul JSON file or directory")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max files to load (0 = all)")
    parser.add_argument("--balance-classes", action="store_true",
                        help="Balance safe/vulnerable classes (DiverseVul only)")
    parser.add_argument("--save-bcell", type=str, default=None,
                        help="Optional path to save B Cell state")
    parser.add_argument("--save-nk", type=str, default=None,
                        help="Optional path to save NK state")
    parser.add_argument("--nk-threshold", type=float, default=0.8,
                        help="NK detection threshold for detector generation")
    parser.add_argument("--nk-detectors", type=int, default=100,
                        help="Number of NK detectors to generate")
    args = parser.parse_args()

    # Load DiverseVul if path provided
    if args.diversevul_path:
        dvul_path = resolve_data_path(args.diversevul_path, None)
        if not dvul_path.exists():
            raise FileNotFoundError(
                f"DiverseVul not found at {dvul_path}\n"
                f"Download from: https://drive.google.com/file/d/12IWKhmLhq7qn5B_iXgn5YerOQtkH-6RG/view"
            )
        safe_samples, vuln_samples = load_diversevul(
            dvul_path,
            max_samples=args.max_samples,
            balance=args.balance_classes
        )
        print(f"Loaded DiverseVul: {len(safe_samples)} safe, {len(vuln_samples)} vulnerable")
        # Train on safe only (NK self = safe code)
        antigens = safe_samples
        eval_samples = vuln_samples
    else:
        # Load local code samples
        data_path = resolve_data_path(args.code_dir, None) if args.code_dir else _code_default_path()
        if not data_path.exists():
            raise FileNotFoundError(f"Code dataset not found at {data_path}")
        antigens = load_code_samples(data_path, max_samples=args.max_samples)
        eval_samples = []

    if not antigens:
        raise ValueError("No code samples found")

    embedder = SimpleTextEmbedder()
    bcell, nk_cell, result = train_code_agents(
        antigens,
        embedder,
        nk_threshold=args.nk_threshold,
        nk_detectors=args.nk_detectors,
    )

    if args.save_bcell:
        bcell.save_state(args.save_bcell)
    if args.save_nk:
        nk_cell.save_state(args.save_nk)

    print("Code training complete")
    print(f"  Files used: {result.files_used}")
    print(f"  B Cell patterns: {result.bcell_patterns}")
    print(f"  NK self patterns: {result.nk_self_patterns}")

    # Quick eval if we have vulnerable samples
    if eval_samples:
        print("\n--- Quick Eval on Vulnerable Samples ---")
        tp, fn = 0, 0
        for ant in eval_samples[:100]:
            emb = embedder.embed(ant.get_text_content())
            result = nk_cell.detect_novelty(ant, antigen_embedding=emb)
            if result.is_anomaly:
                tp += 1
            else:
                fn += 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  Vulnerable detection: {tp}/{tp+fn} ({recall:.1%} recall)")


if __name__ == "__main__":
    main()
