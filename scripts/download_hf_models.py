#!/usr/bin/env python3
"""
Download HuggingFace models for Antigence.

Usage:
    python scripts/download_hf_models.py --all
    python scripts/download_hf_models.py --embeddings
    python scripts/download_hf_models.py --security
    python scripts/download_hf_models.py --model microsoft/codebert-base
"""

import argparse
import os
from pathlib import Path

# Model registry
MODELS = {
    # Embedding models
    "embeddings": [
        {
            "repo_id": "nomic-ai/nomic-embed-text-v1.5",
            "description": "General text embeddings (768 dim)",
            "size_mb": 550,
        },
        {
            "repo_id": "nomic-ai/nomic-embed-code",
            "description": "Code-specific embeddings",
            "size_mb": 550,
        },
        {
            "repo_id": "mixedbread-ai/mxbai-embed-large-v1",
            "description": "High-quality text embeddings (1024 dim)",
            "size_mb": 1340,
        },
    ],
    # Security/vulnerability detection models
    "security": [
        {
            "repo_id": "microsoft/codebert-base",
            "description": "Base CodeBERT for fine-tuning",
            "size_mb": 440,
        },
        {
            "repo_id": "microsoft/graphcodebert-base",
            "description": "Base GraphCodeBERT for fine-tuning",
            "size_mb": 481,
        },
        {
            "repo_id": "mrm8488/codebert-base-finetuned-detect-insecure-code",
            "description": "CodeBERT fine-tuned for insecure code detection",
            "size_mb": 440,
        },
        {
            "repo_id": "mahdin70/GraphCodeBERT-VulnCWE",
            "description": "GraphCodeBERT for vulnerability + CWE classification",
            "size_mb": 481,
        },
        {
            "repo_id": "mahdin70/codebert-devign-code-vulnerability-detector",
            "description": "CodeBERT fine-tuned on Devign dataset",
            "size_mb": 440,
        },
    ],
    # Sentence transformers (alternative embeddings)
    "sentence_transformers": [
        {
            "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "Fast, lightweight embeddings (384 dim)",
            "size_mb": 90,
        },
        {
            "repo_id": "sentence-transformers/all-mpnet-base-v2",
            "description": "High-quality sentence embeddings (768 dim)",
            "size_mb": 420,
        },
    ],
}


def get_cache_dir():
    """Get or create the model cache directory."""
    cache_dir = Path(os.environ.get("ANTIGENCE_MODEL_CACHE", "~/.antigence/models")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_model(repo_id: str, cache_dir: Path, force: bool = False) -> Path:
    """Download a model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / repo_id.replace("/", "_")

    if local_dir.exists() and not force:
        print(f"  [SKIP] {repo_id} already exists at {local_dir}")
        return local_dir

    print(f"  [DOWNLOAD] {repo_id}...")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  [OK] Downloaded to {path}")
        return Path(path)
    except Exception as e:
        print(f"  [ERROR] Failed to download {repo_id}: {e}")
        return None


def list_models():
    """List all available models."""
    print("\n=== Available Models ===\n")
    for category, models in MODELS.items():
        print(f"## {category.upper()}")
        for m in models:
            print(f"  - {m['repo_id']}")
            print(f"    {m['description']} (~{m['size_mb']} MB)")
        print()


def download_category(category: str, cache_dir: Path, force: bool = False):
    """Download all models in a category."""
    if category not in MODELS:
        print(f"Unknown category: {category}")
        return

    models = MODELS[category]
    total_size = sum(m["size_mb"] for m in models)
    print(f"\n=== Downloading {category.upper()} models (~{total_size} MB total) ===\n")

    for m in models:
        download_model(m["repo_id"], cache_dir, force)


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace models for Antigence")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--embeddings", action="store_true", help="Download embedding models")
    parser.add_argument("--security", action="store_true", help="Download security/vuln detection models")
    parser.add_argument("--sentence-transformers", action="store_true", help="Download sentence-transformers")
    parser.add_argument("--model", type=str, help="Download specific model by repo_id")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory")

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else get_cache_dir()
    print(f"Cache directory: {cache_dir}")

    if args.model:
        download_model(args.model, cache_dir, args.force)
    elif args.all:
        for category in MODELS:
            download_category(category, cache_dir, args.force)
    elif args.embeddings:
        download_category("embeddings", cache_dir, args.force)
    elif args.security:
        download_category("security", cache_dir, args.force)
    elif args.sentence_transformers:
        download_category("sentence_transformers", cache_dir, args.force)
    else:
        parser.print_help()
        print("\n[TIP] Use --list to see available models")


if __name__ == "__main__":
    main()
