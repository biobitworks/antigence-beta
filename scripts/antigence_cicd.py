#!/usr/bin/env python3
"""
Antigence CI/CD Manager - Ollama-orchestrated dataset and model management.

This script can be run by local Ollama models to:
1. Check for new datasets
2. Download available datasets
3. Train/retrain models
4. Validate model accuracy
5. Deploy updated models

Usage:
    python scripts/antigence_cicd.py status      # Check current state
    python scripts/antigence_cicd.py download    # Download all available datasets
    python scripts/antigence_cicd.py train       # Train models on downloaded data
    python scripts/antigence_cicd.py validate    # Run validation tests
    python scripts/antigence_cicd.py pipeline    # Run full CI/CD pipeline
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configuration
DATA_DIR = Path.home() / ".antigence" / "data"
MODELS_DIR = Path.home() / ".antigence" / "models"
CONFIG_DIR = Path.home() / ".antigence" / "config"
LOG_FILE = Path.home() / ".antigence" / "cicd.log"

# Datasets that can be auto-downloaded
AUTO_DATASETS = {
    "codexglue": {
        "hf_name": "code_x_glue_cc_defect_detection",
        "description": "Java defect detection (28K samples)",
        "agent": "bcell",
    },
    "truthfulqa": {
        "hf_name": "truthful_qa",
        "hf_subset": "multiple_choice",
        "description": "Truthfulness evaluation (817 questions)",
        "agent": "nkcell",
    },
    "pubmedqa": {
        "hf_name": "pubmed_qa",
        "hf_subset": "pqa_labeled",
        "description": "Medical QA (1K questions)",
        "agent": "bcell",
    },
}

# Datasets requiring manual download
MANUAL_DATASETS = {
    "diversevul": {
        "url": "https://github.com/wagner-group/diversevul.git",
        "command": "git clone https://github.com/wagner-group/diversevul.git",
        "description": "Multi-language vulnerabilities (330K samples)",
    },
    "scifact": {
        "url": "https://github.com/allenai/scifact.git",
        "command": "git clone https://github.com/allenai/scifact.git",
        "description": "Scientific claim verification (1.4K claims)",
    },
    "bigvul": {
        "url": "https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset.git",
        "command": "git clone https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset.git",
        "description": "CVE-linked vulnerabilities (10.9K samples)",
    },
}


def log(message: str):
    """Log message to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")


def check_ollama():
    """Check if Ollama is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_status():
    """Get current CI/CD status."""
    status = {
        "ollama_running": check_ollama(),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "datasets": {},
        "models": {},
    }

    # Check datasets
    for name in AUTO_DATASETS:
        path = DATA_DIR / name
        status["datasets"][name] = {
            "downloaded": path.exists(),
            "path": str(path) if path.exists() else None,
            "type": "auto",
        }

    for name in MANUAL_DATASETS:
        path = DATA_DIR / name
        status["datasets"][name] = {
            "downloaded": path.exists(),
            "path": str(path) if path.exists() else None,
            "type": "manual",
        }

    # Check models
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                status["models"][model_dir.name] = {
                    "path": str(model_dir),
                    "size_mb": sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) // (1024 * 1024),
                }

    return status


def download_hf_dataset(name: str, config: dict) -> bool:
    """Download a HuggingFace dataset."""
    try:
        from datasets import load_dataset

        log(f"Downloading {name}...")

        hf_name = config["hf_name"]
        subset = config.get("hf_subset")

        if subset:
            ds = load_dataset(hf_name, subset)
        else:
            ds = load_dataset(hf_name)

        save_path = DATA_DIR / name
        save_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(save_path))

        # Get stats
        total_samples = sum(len(split) for split in ds.values())
        log(f"  ✓ Downloaded {name}: {total_samples} samples -> {save_path}")
        return True

    except Exception as e:
        log(f"  ✗ Failed to download {name}: {e}")
        return False


def download_all():
    """Download all auto-downloadable datasets."""
    log("=" * 60)
    log("ANTIGENCE CI/CD: Dataset Download")
    log("=" * 60)

    results = {}

    for name, config in AUTO_DATASETS.items():
        path = DATA_DIR / name
        if path.exists():
            log(f"[SKIP] {name} already exists at {path}")
            results[name] = "skipped"
        else:
            success = download_hf_dataset(name, config)
            results[name] = "success" if success else "failed"

    log("-" * 60)
    log("Download Summary:")
    for name, status in results.items():
        log(f"  {name}: {status}")

    return results


def train_models():
    """Train models on downloaded datasets."""
    log("=" * 60)
    log("ANTIGENCE CI/CD: Model Training")
    log("=" * 60)

    # Check what datasets are available
    status = get_status()
    available = [name for name, info in status["datasets"].items() if info["downloaded"]]

    if not available:
        log("No datasets available. Run 'download' first.")
        return {}

    log(f"Available datasets: {', '.join(available)}")

    # Training would happen here
    # For now, just log what would be trained
    results = {}
    for name in available:
        if name in AUTO_DATASETS:
            agent = AUTO_DATASETS[name]["agent"]
            log(f"  Would train {agent} on {name}")
            results[name] = f"ready_for_{agent}"

    return results


def validate_models():
    """Validate trained models."""
    log("=" * 60)
    log("ANTIGENCE CI/CD: Model Validation")
    log("=" * 60)

    # Run the test script
    test_script = Path(__file__).parent / "test_hf_models.py"

    if test_script.exists():
        log(f"Running {test_script}...")
        result = subprocess.run([sys.executable, str(test_script)], capture_output=True, text=True)
        log(result.stdout)
        if result.returncode != 0:
            log(f"Validation failed: {result.stderr}")
            return {"status": "failed", "returncode": result.returncode}
        return {"status": "passed", "returncode": 0}
    else:
        log(f"Test script not found: {test_script}")
        return {"status": "skipped", "reason": "no test script"}


def run_pipeline():
    """Run full CI/CD pipeline."""
    log("=" * 60)
    log("ANTIGENCE CI/CD: Full Pipeline")
    log("=" * 60)

    # Step 1: Check Ollama
    if not check_ollama():
        log("⚠ Ollama not running. Starting...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time
        time.sleep(3)

    # Step 2: Download datasets
    download_results = download_all()

    # Step 3: Train models
    train_results = train_models()

    # Step 4: Validate
    validate_results = validate_models()

    # Summary
    log("=" * 60)
    log("Pipeline Complete")
    log("=" * 60)

    return {
        "downloads": download_results,
        "training": train_results,
        "validation": validate_results,
    }


def print_status():
    """Print current status."""
    status = get_status()

    print("\n" + "=" * 60)
    print("ANTIGENCE CI/CD STATUS")
    print("=" * 60)

    print(f"\nOllama: {'✓ Running' if status['ollama_running'] else '✗ Not running'}")
    print(f"Data Directory: {status['data_dir']}")
    print(f"Models Directory: {status['models_dir']}")

    print("\n--- Datasets ---")
    for name, info in status["datasets"].items():
        icon = "✓" if info["downloaded"] else "○"
        dtype = f"[{info['type']}]"
        print(f"  {icon} {name} {dtype}")

    print("\n--- Models ---")
    if status["models"]:
        for name, info in status["models"].items():
            print(f"  ✓ {name} ({info['size_mb']} MB)")
    else:
        print("  (no models downloaded)")

    print("\n--- Manual Downloads Needed ---")
    for name, info in MANUAL_DATASETS.items():
        path = DATA_DIR / name
        if not path.exists():
            print(f"  {name}: {info['command']}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Antigence CI/CD Manager")
    parser.add_argument("command", choices=["status", "download", "train", "validate", "pipeline"],
                       help="Command to run")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command == "status":
        if args.json:
            print(json.dumps(get_status(), indent=2))
        else:
            print_status()
    elif args.command == "download":
        download_all()
    elif args.command == "train":
        train_models()
    elif args.command == "validate":
        validate_models()
    elif args.command == "pipeline":
        results = run_pipeline()
        if args.json:
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
