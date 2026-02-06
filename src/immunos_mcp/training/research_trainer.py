"""
Train IMMUNOS agents for scientific claim verification using SciFact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # type: ignore

from ..agents.bcell_agent import BCellAgent
from ..agents.nk_cell_agent import NKCellAgent
from ..agents.nk_cell_enhanced import EnhancedNKCellAgent
from ..agents.dendritic_agent import DendriticAgent
from ..agents.memory_agent import MemoryAgent, generate_hash
from ..config.paths import get_data_root, resolve_data_path
from ..core.antigen import Antigen
from ..core.affinity import DistanceMetric
from ..embeddings.simple_text_embedder import SimpleTextEmbedder
from ..embeddings.ollama_embedder import OllamaEmbedder


@dataclass
class ResearchTrainingResult:
    support_count: int
    contradict_count: int
    unknown_count: int
    bcell_patterns: int
    nk_self_patterns: int


def _scifact_default_path() -> Path:
    candidates = [
        get_data_root() / "research" / "scifact" / "data" / "claims_train.jsonl",
        get_data_root() / "research" / "scifact" / "data" / "claims_dev.jsonl",
        get_data_root() / "research" / "scifact" / "claims_train.jsonl",
        get_data_root() / "research" / "scifact" / "claims_dev.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _derive_label(evidence: dict) -> str:
    labels = set()
    for rationales in evidence.values():
        for entry in rationales:
            label = entry.get("label")
            if label:
                labels.add(label.upper())
    if "CONTRADICT" in labels:
        return "contradict"
    if "SUPPORT" in labels:
        return "support"
    return "unknown"


def _scifact_corpus_default_path() -> Path:
    candidates = [
        get_data_root() / "research" / "scifact" / "data" / "corpus.jsonl",
        get_data_root() / "research" / "scifact" / "corpus.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


_SCIFACT_CORPUS_CACHE: Dict[str, Dict[str, List[str]]] = {}
_SENTENCE_EMBED_CACHE: Dict[str, np.ndarray] = {}


def _load_scifact_corpus(path: Path) -> Dict[str, List[str]]:
    cache_key = str(path)
    cached = _SCIFACT_CORPUS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    corpus: Dict[str, List[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            entry = json.loads(raw)
            doc_id = str(entry.get("doc_id"))
            abstract = entry.get("abstract") or []
            if doc_id:
                corpus[doc_id] = abstract

    _SCIFACT_CORPUS_CACHE[cache_key] = corpus
    return corpus


def _extract_evidence_sentences(evidence: dict,
                                corpus: Dict[str, List[str]],
                                max_sentences: int) -> Tuple[List[str], List[Tuple[str, int]]]:
    collected: List[str] = []
    indices: List[Tuple[str, int]] = []
    for doc_id, rationales in evidence.items():
        doc_sentences = corpus.get(str(doc_id))
        if not doc_sentences:
            continue
        for rationale in rationales:
            for sent_idx in rationale.get("sentences", []):
                if 0 <= sent_idx < len(doc_sentences):
                    collected.append(doc_sentences[sent_idx])
                    indices.append((str(doc_id), sent_idx))
                    if len(collected) >= max_sentences:
                        return collected, indices
    return collected, indices


def _retrieve_evidence_sentences(claim_text: str,
                                 cited_doc_ids: List[int],
                                 corpus: Dict[str, List[str]],
                                 embedder: Any,
                                 max_sentences: int,
                                 max_candidates_per_doc: int) -> Tuple[List[str], List[Tuple[str, int]]]:
    if not cited_doc_ids:
        return [], []

    candidates: List[Tuple[str, str, int]] = []
    for doc_id in cited_doc_ids:
        doc_sentences = corpus.get(str(doc_id))
        if not doc_sentences:
            continue
        for idx, sentence in enumerate(doc_sentences[:max_candidates_per_doc]):
            candidates.append((sentence, str(doc_id), idx))

    if not candidates:
        return [], []

    claim_embedding = embedder.embed(claim_text)
    scored: List[Tuple[float, str, str, int]] = []
    for sentence, doc_id, idx in candidates:
        cached = _SENTENCE_EMBED_CACHE.get(sentence)
        if cached is None:
            cached = embedder.embed(sentence)
            _SENTENCE_EMBED_CACHE[sentence] = cached
        similarity = 1.0 - DistanceMetric.cosine_distance(claim_embedding, cached)
        scored.append((similarity, sentence, doc_id, idx))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[:max_sentences]
    sentences = [item[1] for item in selected]
    indices = [(item[2], item[3]) for item in selected]
    return sentences, indices


def apply_retrieved_evidence(antigens: List[Antigen],
                             embedder: Any,
                             corpus_path: Optional[Path] = None,
                             max_evidence_sentences: int = 3,
                             max_candidates_per_doc: int = 8) -> List[Antigen]:
    corpus_path = corpus_path or _scifact_corpus_default_path()
    if not corpus_path.exists():
        print(f"[WARN] SciFact corpus not found at {corpus_path}; retrieval skipped.")
        return antigens

    corpus = _load_scifact_corpus(corpus_path)
    for antigen in antigens:
        claim_text = antigen.metadata.get("claim_text", antigen.get_text_content())
        cited_doc_ids = antigen.metadata.get("cited_doc_ids", [])
        sentences, indices = _retrieve_evidence_sentences(
            claim_text,
            cited_doc_ids,
            corpus,
            embedder,
            max_evidence_sentences,
            max_candidates_per_doc,
        )
        if sentences:
            antigen.data = f"{claim_text}\nEvidence: " + " ".join(sentences)
            antigen.metadata["retrieved_evidence_sentences"] = sentences
            antigen.metadata["retrieved_evidence_sentence_ids"] = indices
    return antigens


def load_scifact_claims(path: Path,
                        max_samples: int = 0,
                        include_evidence: bool = False,
                        corpus_path: Optional[Path] = None,
                        max_evidence_sentences: int = 3) -> Tuple[List[Antigen], List[str]]:
    antigens: List[Antigen] = []
    labels: List[str] = []
    corpus: Optional[Dict[str, List[str]]] = None

    if include_evidence:
        corpus_path = corpus_path or _scifact_corpus_default_path()
        if corpus_path.exists():
            corpus = _load_scifact_corpus(corpus_path)
        else:
            print(f"[WARN] SciFact corpus not found at {corpus_path}; evidence will be skipped.")

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            claim_text = entry.get("claim", "").strip()
            if not claim_text:
                continue

            label = _derive_label(entry.get("evidence", {}))
            evidence_sentences: List[str] = []
            evidence_indices: List[Tuple[str, int]] = []
            if include_evidence and corpus:
                evidence_sentences, evidence_indices = _extract_evidence_sentences(
                    entry.get("evidence", {}),
                    corpus,
                    max_evidence_sentences,
                )
                if evidence_sentences:
                    claim_text = f"{claim_text}\nEvidence: " + " ".join(evidence_sentences)
            antigen = Antigen.from_text(
                claim_text,
                class_label=label,
                metadata={
                    "claim_id": entry.get("id"),
                    "claim_text": entry.get("claim", "").strip(),
                    "evidence_doc_ids": list(entry.get("evidence", {}).keys()),
                    "cited_doc_ids": entry.get("cited_doc_ids", []),
                    "evidence_sentences": evidence_sentences,
                    "evidence_sentence_ids": evidence_indices,
                }
            )
            antigens.append(antigen)
            labels.append(label)

            if max_samples and len(antigens) >= max_samples:
                break

    return antigens, labels


def _build_embedder(name: str, ollama_model: Optional[str], ollama_base_url: Optional[str]):
    if name == "ollama":
        return OllamaEmbedder(
            model=ollama_model or "nomic-embed-text",
            base_url=ollama_base_url
        )
    return SimpleTextEmbedder()


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_info(repo_root: Path) -> Dict[str, Optional[str]]:
    info = {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["commit"] = commit
        info["dirty"] = "yes" if status else "no"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return info
    return info


def _list_dependencies() -> List[str]:
    deps = []
    try:
        for dist in importlib_metadata.distributions():
            name = dist.metadata.get("Name")
            version = dist.version
            if name and version:
                deps.append(f"{name}=={version}")
    except Exception:
        return deps
    return sorted(deps, key=lambda item: item.lower())


def _lockfiles(repo_root: Path) -> List[str]:
    candidates = [
        "pyproject.toml",
        "uv.lock",
        "poetry.lock",
        "requirements.txt",
        "environment.yml",
        "environment.yaml",
    ]
    found = []
    for name in candidates:
        path = repo_root / name
        if path.exists():
            found.append(str(path))
    return found


def _environment_snapshot(repo_root: Path) -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "data_root": os.environ.get("IMMUNOS_DATA_ROOT"),
        "lockfiles": _lockfiles(repo_root),
        "dependencies": _list_dependencies(),
        "git": _git_info(repo_root),
    }


def _agent_signature() -> Optional[Dict[str, Optional[str]]]:
    agent = {
        "agent_type": os.environ.get("IMMUNOS_AGENT_TYPE"),
        "agent_id": os.environ.get("IMMUNOS_AGENT_ID"),
        "session_id": os.environ.get("IMMUNOS_SESSION_ID"),
        "key_id": os.environ.get("IMMUNOS_KEY_ID"),
        "trust": os.environ.get("IMMUNOS_TRUST"),
        "model": os.environ.get("IMMUNOS_MODEL"),
        "signature": os.environ.get("IMMUNOS_AGENT_SIGNATURE"),
        "status": os.environ.get("IMMUNOS_AGENT_STATUS"),
        "started_at": os.environ.get("IMMUNOS_SESSION_STARTED_AT"),
    }
    if not any(agent.values()):
        return None
    return agent


def _unique_manifest_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(1, 100):
        candidate = path.with_name(f"{stem}-{idx}{suffix}")
        if not candidate.exists():
            return candidate
    return path


def _build_scifact_manifest(data_path: Path,
                            corpus_path: Optional[Path],
                            args: argparse.Namespace,
                            result: ResearchTrainingResult) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    corpus_real = corpus_path if corpus_path and corpus_path.exists() else None
    manifest = {
        "dataset": "scifact",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "repo_root": str(repo_root),
        "claims_path": str(data_path),
        "claims_sha256": _sha256_file(data_path),
        "corpus_path": str(corpus_real) if corpus_real else None,
        "corpus_sha256": _sha256_file(corpus_real) if corpus_real else None,
        "evidence_mode": args.evidence_mode or ("gold" if args.include_evidence else "none"),
        "evidence_max": args.evidence_max,
        "evidence_candidates_per_doc": args.evidence_candidates_per_doc,
        "max_samples": args.max_samples,
        "include_unknown": args.include_unknown,
        "balance_classes": args.balance_classes,
        "seed": args.seed,
        "embedder": args.embedder,
        "ollama_model": args.ollama_model if args.embedder == "ollama" else None,
        "ollama_base_url": args.ollama_base_url if args.embedder == "ollama" else None,
        "bcell_affinity": args.bcell_affinity,
        "bcell_embedding_weight": args.bcell_embedding_weight,
        "nk_threshold": args.nk_threshold,
        "nk_detectors": args.nk_detectors,
        "nk_mode": args.nk_mode,
        "nk_threshold_method": args.nk_threshold_method if args.nk_mode == "enhanced" else None,
        "nk_detectors_per_class": args.nk_detectors_per_class if args.nk_mode == "enhanced" else None,
        "support_count": result.support_count,
        "contradict_count": result.contradict_count,
        "unknown_count": result.unknown_count,
        "bcell_patterns": result.bcell_patterns,
        "nk_self_patterns": result.nk_self_patterns,
        "outputs": {
            "bcell_state": args.save_bcell,
            "nk_state": args.save_nk,
            "manifest_path": args.write_manifest,
        },
        "environment": _environment_snapshot(repo_root),
        "agent": _agent_signature(),
    }
    return manifest


def train_research_agents(antigens: List[Antigen],
                          labels: List[str],
                          embedder: Any,
                          nk_threshold: float,
                          nk_detectors: int,
                          nk_threshold_method: str,
                          nk_detectors_per_class: int,
                          include_unknown: bool,
                          balance_classes: bool = False,
                          seed: int = 42,
                          bcell_affinity: str = "hybrid",
                          bcell_embedding_weight: float = 0.7,
                          use_dendritic: bool = False,
                          use_memory: bool = False,
                          memory_path: Optional[Path] = None,
                          nk_mode: str = "embedding") -> Tuple[BCellAgent, NKCellAgent, ResearchTrainingResult]:
    """
    Train B Cell and NK Cell agents for research claim verification.

    Args:
        nk_mode: "embedding" for dense embeddings (1024-dim), "feature" for Dendritic features (20-dim)
    """
    bcell = BCellAgent(
        agent_name="research_bcell",
        affinity_method=bcell_affinity,
        embedding_weight=bcell_embedding_weight,
    )
    if nk_mode == "enhanced":
        nk_cell = EnhancedNKCellAgent(
            agent_name="research_nk_enhanced",
            threshold_method=nk_threshold_method,
            detectors_per_class=nk_detectors_per_class,
        )
    else:
        nk_cell = NKCellAgent(
            agent_name="research_nk",
            detection_threshold=nk_threshold,
            num_detectors=nk_detectors,
            mode=nk_mode,
        )

    # Force Dendritic extraction if using feature-based NK
    if nk_mode == "feature":
        use_dendritic = True

    # Initialize Dendritic and Memory agents if enabled
    dendritic = DendriticAgent() if use_dendritic else None
    memory = None
    if use_memory:
        mem_path = memory_path or Path(".immunos") / "memory" / "research_memory.json"
        memory = MemoryAgent(path=mem_path, max_memories=10000)

    # Extract Dendritic features for all antigens
    if dendritic:
        print(f"[research_trainer] Extracting Dendritic features for {len(antigens)} antigens...")
        for antigen in antigens:
            features = dendritic.extract_features(antigen)
            antigen.features = features

    filtered_pairs: List[Tuple[Antigen, str]] = []
    for antigen, label in zip(antigens, labels):
        if label == "unknown" and not include_unknown:
            continue
        filtered_pairs.append((antigen, label))

    if balance_classes and filtered_pairs:
        rng = random.Random(seed)
        by_label: Dict[str, List[Tuple[Antigen, str]]] = defaultdict(list)
        for antigen, label in filtered_pairs:
            by_label[label].append((antigen, label))
        min_count = min(len(items) for items in by_label.values())
        balanced: List[Tuple[Antigen, str]] = []
        for label, items in by_label.items():
            rng.shuffle(items)
            balanced.extend(items[:min_count])
        rng.shuffle(balanced)
        filtered_pairs = balanced

    filtered_antigens = [pair[0] for pair in filtered_pairs]
    filtered_labels = [pair[1] for pair in filtered_pairs]

    embeddings = [embedder.embed(antigen.get_text_content()) for antigen in filtered_antigens]
    bcell.train(filtered_antigens, embeddings=embeddings)

    # Train NK Cell on "self" (support claims)
    support_antigens = [antigen for antigen in filtered_antigens if antigen.class_label == "support"]

    if nk_mode == "feature" and dendritic:
        # Feature-based NK: use Dendritic feature vectors (20-dim)
        support_features = [dendritic.get_feature_vector(antigen) for antigen in support_antigens]
        nk_cell.train_on_features(support_antigens, support_features)
    elif nk_mode == "enhanced":
        # Enhanced NK: use dense embeddings (one-vs-rest optional)
        support_embeddings = [embedder.embed(antigen.get_text_content()) for antigen in support_antigens]
        nk_cell.train_on_self(support_antigens, embeddings=support_embeddings)
    else:
        # Embedding-based NK: use dense embeddings (1024-dim)
        support_embeddings = [embedder.embed(antigen.get_text_content()) for antigen in support_antigens]
        nk_cell.train_on_self(support_antigens, embeddings=support_embeddings)

    support_count = filtered_labels.count("support")
    contradict_count = filtered_labels.count("contradict")
    unknown_count = filtered_labels.count("unknown")

    # Store training samples in Memory for few-shot learning
    if memory:
        print(f"[research_trainer] Storing {len(filtered_antigens)} samples in Memory...")
        for antigen, embedding in zip(filtered_antigens, embeddings):
            memory.store_classification(
                antigen_hash=generate_hash(antigen.get_text_content()),
                antigen_text=antigen.get_text_content(),
                class_label=antigen.class_label,
                confidence=0.9,  # Training samples have high confidence
                embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                features=antigen.features if hasattr(antigen, 'features') else None
            )
        print(f"[research_trainer] Memory stats: {memory.get_statistics()}")

    if nk_mode == "enhanced":
        nk_self_patterns = nk_cell.training_stats.get("total_self_patterns", 0)
    else:
        nk_self_patterns = len(nk_cell.self_patterns)

    result = ResearchTrainingResult(
        support_count=support_count,
        contradict_count=contradict_count,
        unknown_count=unknown_count,
        bcell_patterns=len(bcell.patterns),
        nk_self_patterns=nk_self_patterns,
    )
    return bcell, nk_cell, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IMMUNOS research agents (SciFact)")
    parser.add_argument("--claims", type=str, default=None,
                        help="Path to SciFact claims JSONL (absolute or relative to data root)")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Max claims to load (0 = all)")
    parser.add_argument("--include-unknown", action="store_true",
                        help="Include unknown claims in B Cell training")
    parser.add_argument("--balance-classes", action="store_true",
                        help="Downsample labels to the smallest class size for balanced training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for balanced sampling")
    parser.add_argument("--include-evidence", action="store_true",
                        help="Append gold evidence sentences from the SciFact corpus to claims")
    parser.add_argument("--evidence-mode", type=str, default=None,
                        choices=["gold", "retrieval", "none"],
                        help="Evidence augmentation mode (default: gold if --include-evidence)")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Path to SciFact corpus.jsonl (absolute or relative to data root)")
    parser.add_argument("--evidence-max", type=int, default=3,
                        help="Max evidence sentences to append per claim")
    parser.add_argument("--evidence-candidates-per-doc", type=int, default=8,
                        help="Max sentences per cited document to consider for retrieval")
    parser.add_argument("--embedder", type=str, default="simple_text",
                        choices=["simple_text", "ollama"],
                        help="Embedding backend for research training")
    parser.add_argument("--bcell-affinity", type=str, default="hybrid",
                        choices=["traditional", "embedding", "hybrid"],
                        help="B Cell affinity method")
    parser.add_argument("--bcell-embedding-weight", type=float, default=0.7,
                        help="Embedding weight for hybrid affinity (0-1)")
    parser.add_argument("--ollama-model", type=str, default="nomic-embed-text",
                        help="Ollama embedding model name")
    parser.add_argument("--ollama-base-url", type=str, default=None,
                        help="Override Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--nk-threshold", type=float, default=0.85,
                        help="NK detection threshold for detector generation")
    parser.add_argument("--nk-detectors", type=int, default=100,
                        help="Number of NK detectors to generate")
    parser.add_argument("--nk-mode", type=str, default="embedding",
                        choices=["embedding", "feature", "enhanced"],
                        help="NK mode: embedding (1024-dim), feature (20-dim Dendritic), or enhanced (per-class)")
    parser.add_argument("--nk-threshold-method", type=str, default="min_distance",
                        choices=["min_distance", "mean_distance", "percentile"],
                        help="Enhanced NK threshold method")
    parser.add_argument("--nk-detectors-per-class", type=int, default=25,
                        help="Enhanced NK detectors per class")
    parser.add_argument("--write-manifest", type=str, default=None,
                        help="Optional path to write a SciFact training manifest (JSON)")
    parser.add_argument("--save-bcell", type=str, default=None,
                        help="Optional path to save B Cell state")
    parser.add_argument("--save-nk", type=str, default=None,
                        help="Optional path to save NK state")
    parser.add_argument("--use-dendritic", action="store_true",
                        help="Extract Dendritic features for each antigen")
    parser.add_argument("--use-memory", action="store_true",
                        help="Store training samples in Memory for few-shot learning")
    parser.add_argument("--memory-path", type=str, default=None,
                        help="Path to Memory storage file")
    args = parser.parse_args()

    data_path = resolve_data_path(args.claims, None) if args.claims else _scifact_default_path()
    if not data_path.exists():
        raise FileNotFoundError(
            f"SciFact claims not found at {data_path}. "
            "Run `research/scifact/script/download-data.sh` or download data.tar.gz."
        )

    corpus_path = resolve_data_path(args.corpus, None) if args.corpus else None
    evidence_mode = args.evidence_mode or ("gold" if args.include_evidence else "none")
    include_gold = evidence_mode == "gold"
    antigens, labels = load_scifact_claims(
        data_path,
        max_samples=args.max_samples,
        include_evidence=include_gold,
        corpus_path=corpus_path,
        max_evidence_sentences=args.evidence_max,
    )
    embedder = _build_embedder(args.embedder, args.ollama_model, args.ollama_base_url)

    if evidence_mode == "retrieval":
        antigens = apply_retrieved_evidence(
            antigens,
            embedder,
            corpus_path=corpus_path,
            max_evidence_sentences=args.evidence_max,
            max_candidates_per_doc=args.evidence_candidates_per_doc,
        )

    memory_path = Path(args.memory_path) if args.memory_path else None
    bcell, nk_cell, result = train_research_agents(
        antigens,
        labels,
        embedder,
        nk_threshold=args.nk_threshold,
        nk_detectors=args.nk_detectors,
        nk_threshold_method=args.nk_threshold_method,
        nk_detectors_per_class=args.nk_detectors_per_class,
        include_unknown=args.include_unknown,
        balance_classes=args.balance_classes,
        seed=args.seed,
        bcell_affinity=args.bcell_affinity,
        bcell_embedding_weight=args.bcell_embedding_weight,
        use_dendritic=args.use_dendritic,
        use_memory=args.use_memory,
        memory_path=memory_path,
        nk_mode=args.nk_mode,
    )

    if args.save_bcell:
        bcell.save_state(args.save_bcell)
    if args.save_nk:
        nk_cell.save_state(args.save_nk)

    if args.write_manifest:
        manifest_path = _unique_manifest_path(Path(args.write_manifest))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = _build_scifact_manifest(data_path, corpus_path, args, result)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("✅ Research training complete")
    print(f"  Support claims: {result.support_count}")
    print(f"  Contradict claims: {result.contradict_count}")
    print(f"  Unknown claims: {result.unknown_count}")
    print(f"  B Cell patterns: {result.bcell_patterns}")
    print(f"  NK self patterns: {result.nk_self_patterns}")


if __name__ == "__main__":
    main()
