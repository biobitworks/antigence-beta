#!/usr/bin/env python3
"""
Download training datasets for Antigence.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --scifact
    python scripts/download_datasets.py --diversevul
    python scripts/download_datasets.py --list
"""

import argparse
import os
from pathlib import Path

# Dataset registry
DATASETS = {
    "scifact": {
        "hf_repo": "allenai/scifact",
        "description": "Scientific claim verification (1.4K claims)",
        "size_mb": 50,
        "use_case": "B Cell claim verification training",
    },
    "fever": {
        "hf_repo": "fever/fever",
        "description": "Fact verification dataset (185K claims)",
        "size_mb": 500,
        "use_case": "Hallucination detection training",
    },
    "truthful_qa": {
        "hf_repo": "truthful_qa",
        "description": "Truthfulness benchmark (817 questions)",
        "size_mb": 10,
        "use_case": "Hallucination detection evaluation",
    },
    "devign": {
        "url": "https://raw.githubusercontent.com/epicosy/devign/main/data/function.json",
        "description": "Real-world C vulnerabilities from FFmpeg/Qemu",
        "size_mb": 150,
        "use_case": "Code vulnerability training",
    },
}


def get_data_dir():
    """Get or create the data directory."""
    data_dir = Path(os.environ.get("ANTIGENCE_DATA_DIR", "~/.antigence/data")).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_hf_dataset(repo_id: str, data_dir: Path, name: str) -> Path:
    """Download a dataset from HuggingFace."""
    from datasets import load_dataset

    local_dir = data_dir / name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [DOWNLOAD] {repo_id}...")
    try:
        dataset = load_dataset(repo_id)

        # Save to disk
        dataset.save_to_disk(str(local_dir))
        print(f"  [OK] Downloaded to {local_dir}")

        # Print stats
        for split_name, split_data in dataset.items():
            print(f"       {split_name}: {len(split_data)} examples")

        return local_dir
    except Exception as e:
        print(f"  [ERROR] Failed to download {repo_id}: {e}")
        return None


def download_url(url: str, data_dir: Path, name: str) -> Path:
    """Download a file from URL."""
    import requests

    local_path = data_dir / f"{name}.json"

    print(f"  [DOWNLOAD] {url}...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with open(local_path, "w") as f:
            f.write(response.text)

        print(f"  [OK] Downloaded to {local_path}")
        return local_path
    except Exception as e:
        print(f"  [ERROR] Failed to download: {e}")
        return None


def list_datasets():
    """List all available datasets."""
    print("\n=== Available Datasets ===\n")
    for name, info in DATASETS.items():
        source = info.get("hf_repo") or info.get("url", "manual")
        print(f"  {name}")
        print(f"    Description: {info['description']}")
        print(f"    Size: ~{info['size_mb']} MB")
        print(f"    Use case: {info['use_case']}")
        print(f"    Source: {source}")
        print()


def download_dataset(name: str, data_dir: Path) -> Path:
    """Download a specific dataset."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        return None

    info = DATASETS[name]

    if "hf_repo" in info:
        return download_hf_dataset(info["hf_repo"], data_dir, name)
    elif "url" in info:
        return download_url(info["url"], data_dir, name)
    else:
        print(f"  [SKIP] {name} requires manual download")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download training datasets for Antigence")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--scifact", action="store_true", help="Download SciFact dataset")
    parser.add_argument("--fever", action="store_true", help="Download FEVER dataset")
    parser.add_argument("--truthful-qa", action="store_true", help="Download TruthfulQA")
    parser.add_argument("--devign", action="store_true", help="Download Devign dataset")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--data-dir", type=str, help="Custom data directory")

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else get_data_dir()
    print(f"Data directory: {data_dir}")

    if args.all:
        for name in DATASETS:
            download_dataset(name, data_dir)
    elif args.scifact:
        download_dataset("scifact", data_dir)
    elif args.fever:
        download_dataset("fever", data_dir)
    elif args.truthful_qa:
        download_dataset("truthful_qa", data_dir)
    elif args.devign:
        download_dataset("devign", data_dir)
    else:
        parser.print_help()
        print("\n[TIP] Use --list to see available datasets")


if __name__ == "__main__":
    main()
