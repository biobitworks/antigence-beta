#!/usr/bin/env python3
"""
Train Antigence agents on downloaded datasets.

Usage:
    python scripts/train_agents.py --bcell        # Train B Cell only
    python scripts/train_agents.py --nkcell       # Train NK Cell only
    python scripts/train_agents.py --all          # Train all agents
    python scripts/train_agents.py --status       # Show training data status
"""

import argparse
import json
import csv
import random
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data"
MODELS_DIR = Path.home() / ".antigence" / "trained"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_bigvul(max_samples: int = 10000) -> list:
    """Load Big-Vul CVE dataset."""
    csv_path = DATA_DIR / "bigvul" / "all_c_cpp_release2.0.csv"
    if not csv_path.exists():
        log(f"Big-Vul not found at {csv_path}")
        return []

    log(f"Loading Big-Vul from {csv_path}...")
    samples = []

    # Increase CSV field size limit for large patches
    import sys
    csv.field_size_limit(sys.maxsize)

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_samples * 2:  # Load more, filter later
                break

            # Extract code from patch if available
            patch = row.get("files_changed", "")
            cwe = row.get("cwe_id", "")
            cve = row.get("cve_id", "")

            if patch and cwe:
                samples.append({
                    "code": patch[:2000],  # Truncate long patches
                    "label": "vulnerable",
                    "cwe": cwe,
                    "cve": cve,
                    "source": "bigvul",
                })

    # Balance with safe samples (use first half of code as "safe" proxy)
    safe_samples = []
    for s in samples[:len(samples)//2]:
        safe_samples.append({
            "code": s["code"][:500],  # Shorter = less likely vulnerable
            "label": "safe",
            "source": "bigvul_safe_proxy",
        })

    all_samples = samples[:max_samples//2] + safe_samples[:max_samples//2]
    random.shuffle(all_samples)

    log(f"  Loaded {len(all_samples)} samples (vuln + safe proxy)")
    return all_samples


def load_codexglue() -> list:
    """Load CodeXGLUE defect detection dataset."""
    ds_path = DATA_DIR / "codexglue"
    if not ds_path.exists():
        log(f"CodeXGLUE not found at {ds_path}")
        return []

    log(f"Loading CodeXGLUE from {ds_path}...")

    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(ds_path))

        samples = []
        for split in ["train", "validation", "test"]:
            if split in ds:
                for item in ds[split]:
                    samples.append({
                        "code": item.get("func", "")[:2000],
                        "label": "vulnerable" if item.get("target", 0) == 1 else "safe",
                        "source": "codexglue",
                    })

        log(f"  Loaded {len(samples)} samples")
        return samples

    except Exception as e:
        log(f"  Error loading CodeXGLUE: {e}")
        return []


def load_truthfulqa() -> list:
    """Load TruthfulQA for hallucination detection."""
    ds_path = DATA_DIR / "truthfulqa"
    if not ds_path.exists():
        log(f"TruthfulQA not found at {ds_path}")
        return []

    log(f"Loading TruthfulQA from {ds_path}...")

    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(ds_path))

        samples = []
        for split in ds:
            for item in ds[split]:
                question = item.get("question", "")
                # mc1_targets has correct answers
                mc1 = item.get("mc1_targets", {})
                choices = mc1.get("choices", [])
                labels = mc1.get("labels", [])

                for choice, label in zip(choices, labels):
                    samples.append({
                        "text": f"Q: {question}\nA: {choice}",
                        "label": "truthful" if label == 1 else "hallucinated",
                        "source": "truthfulqa",
                    })

        log(f"  Loaded {len(samples)} samples")
        return samples

    except Exception as e:
        log(f"  Error loading TruthfulQA: {e}")
        return []


def load_scifact() -> list:
    """Load SciFact claims."""
    scifact_dir = DATA_DIR / "scifact"
    claims_file = scifact_dir / "claims_train.jsonl"

    if not claims_file.exists():
        # Try alternative location
        claims_file = scifact_dir / "data" / "claims_train.jsonl"

    if not claims_file.exists():
        log(f"SciFact not found")
        return []

    log(f"Loading SciFact from {claims_file}...")
    samples = []

    try:
        with open(claims_file, "r") as f:
            for line in f:
                item = json.loads(line)
                samples.append({
                    "text": item.get("claim", ""),
                    "label": "supported" if item.get("evidence") else "unsupported",
                    "source": "scifact",
                })

        log(f"  Loaded {len(samples)} samples")
        return samples

    except Exception as e:
        log(f"  Error loading SciFact: {e}")
        return []


def train_bcell(samples: list, name: str = "bcell_security"):
    """Train B Cell agent on samples."""
    log(f"\n{'='*60}")
    log(f"Training B Cell: {name}")
    log(f"{'='*60}")

    if not samples:
        log("No samples to train on!")
        return None

    # Split into train/test
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_data = samples[:split]
    test_data = samples[split:]

    log(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Count labels
    label_counts = {}
    for s in train_data:
        label = s.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    log(f"Label distribution: {label_counts}")

    # Save training data
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / f"{name}_training.json"

    with open(output_path, "w") as f:
        json.dump({
            "name": name,
            "trained_at": datetime.now().isoformat(),
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "label_counts": label_counts,
            "train_data": train_data[:100],  # Save sample for reference
        }, f, indent=2)

    log(f"Saved training config to {output_path}")

    # Here we would actually train the model
    # For now, create embeddings and patterns

    try:
        import requests

        log("Generating embeddings via Ollama...")
        patterns = []

        for i, sample in enumerate(train_data[:500]):  # Limit for speed
            if i % 100 == 0:
                log(f"  Processing {i}/{min(500, len(train_data))}...")

            text = sample.get("code") or sample.get("text", "")
            if not text:
                continue

            # Get embedding
            try:
                resp = requests.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": text[:1000]},
                    timeout=30
                )
                if resp.status_code == 200:
                    embedding = resp.json().get("embedding", [])
                    patterns.append({
                        "label": sample["label"],
                        "embedding": embedding[:100],  # Truncate for storage
                        "source": sample.get("source", "unknown"),
                    })
            except:
                pass

        # Save patterns
        patterns_path = MODELS_DIR / f"{name}_patterns.json"
        with open(patterns_path, "w") as f:
            json.dump({
                "name": name,
                "created": datetime.now().isoformat(),
                "num_patterns": len(patterns),
                "patterns": patterns,
            }, f)

        log(f"Saved {len(patterns)} patterns to {patterns_path}")
        return patterns_path

    except Exception as e:
        log(f"Error generating embeddings: {e}")
        return None


def train_nkcell(samples: list, name: str = "nkcell_hallucination"):
    """Train NK Cell agent (self patterns only)."""
    log(f"\n{'='*60}")
    log(f"Training NK Cell: {name}")
    log(f"{'='*60}")

    if not samples:
        log("No samples to train on!")
        return None

    # NK Cell only trains on "self" (truthful/safe samples)
    self_samples = [s for s in samples if s.get("label") in ["truthful", "safe", "supported"]]
    log(f"Self samples: {len(self_samples)} (from {len(samples)} total)")

    if not self_samples:
        log("No self samples found!")
        return None

    # Save self patterns
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / f"{name}_self.json"

    try:
        import requests

        log("Generating self-pattern embeddings via Ollama...")
        self_patterns = []

        for i, sample in enumerate(self_samples[:200]):  # Limit for speed
            if i % 50 == 0:
                log(f"  Processing {i}/{min(200, len(self_samples))}...")

            text = sample.get("text") or sample.get("code", "")
            if not text:
                continue

            try:
                resp = requests.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": text[:1000]},
                    timeout=30
                )
                if resp.status_code == 200:
                    embedding = resp.json().get("embedding", [])
                    self_patterns.append({
                        "embedding": embedding[:100],
                        "source": sample.get("source", "unknown"),
                    })
            except:
                pass

        with open(output_path, "w") as f:
            json.dump({
                "name": name,
                "created": datetime.now().isoformat(),
                "num_self_patterns": len(self_patterns),
                "self_patterns": self_patterns,
            }, f)

        log(f"Saved {len(self_patterns)} self-patterns to {output_path}")
        return output_path

    except Exception as e:
        log(f"Error: {e}")
        return None


def show_status():
    """Show available training data."""
    log("\n" + "="*60)
    log("TRAINING DATA STATUS")
    log("="*60)

    datasets = {
        "Big-Vul": DATA_DIR / "bigvul" / "all_c_cpp_release2.0.csv",
        "CodeXGLUE": DATA_DIR / "codexglue",
        "TruthfulQA": DATA_DIR / "truthfulqa",
        "SciFact": DATA_DIR / "scifact",
        "PubMedQA": DATA_DIR / "pubmedqa",
    }

    for name, path in datasets.items():
        exists = path.exists()
        status = "✓" if exists else "○"
        print(f"  {status} {name}: {path}")

    print("\n--- Trained Models ---")
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.json"):
            print(f"  ✓ {f.name}")
    else:
        print("  (none)")


def main():
    parser = argparse.ArgumentParser(description="Train Antigence agents")
    parser.add_argument("--bcell", action="store_true", help="Train B Cell")
    parser.add_argument("--nkcell", action="store_true", help="Train NK Cell")
    parser.add_argument("--all", action="store_true", help="Train all agents")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples per dataset")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.bcell or args.all:
        # Load code security datasets
        samples = []
        samples.extend(load_bigvul(args.max_samples))
        samples.extend(load_codexglue())

        if samples:
            train_bcell(samples, "bcell_code_security")

    if args.nkcell or args.all:
        # Load truthfulness datasets
        samples = load_truthfulqa()
        if samples:
            train_nkcell(samples, "nkcell_truthfulness")

    log("\n" + "="*60)
    log("Training Complete!")
    log("="*60)
    show_status()


if __name__ == "__main__":
    main()
